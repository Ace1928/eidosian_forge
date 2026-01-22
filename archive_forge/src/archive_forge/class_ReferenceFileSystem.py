import base64
import collections
import io
import itertools
import logging
import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING
import fsspec.core
from ..asyn import AsyncFileSystem
from ..callbacks import DEFAULT_CALLBACK
from ..core import filesystem, open, split_protocol
from ..utils import isfilelike, merge_offset_ranges, other_paths
class ReferenceFileSystem(AsyncFileSystem):
    """View byte ranges of some other file as a file system
    Initial version: single file system target, which must support
    async, and must allow start and end args in _cat_file. Later versions
    may allow multiple arbitrary URLs for the targets.
    This FileSystem is read-only. It is designed to be used with async
    targets (for now). This FileSystem only allows whole-file access, no
    ``open``. We do not get original file details from the target FS.
    Configuration is by passing a dict of references at init, or a URL to
    a JSON file containing the same; this dict
    can also contain concrete data for some set of paths.
    Reference dict format:
    {path0: bytes_data, path1: (target_url, offset, size)}
    https://github.com/fsspec/kerchunk/blob/main/README.md
    """
    protocol = 'reference'

    def __init__(self, fo, target=None, ref_storage_args=None, target_protocol=None, target_options=None, remote_protocol=None, remote_options=None, fs=None, template_overrides=None, simple_templates=True, max_gap=64000, max_block=256000000, cache_size=128, **kwargs):
        """
        Parameters
        ----------
        fo : dict or str
            The set of references to use for this instance, with a structure as above.
            If str referencing a JSON file, will use fsspec.open, in conjunction
            with target_options and target_protocol to open and parse JSON at this
            location. If a directory, then assume references are a set of parquet
            files to be loaded lazily.
        target : str
            For any references having target_url as None, this is the default file
            target to use
        ref_storage_args : dict
            If references is a str, use these kwargs for loading the JSON file.
            Deprecated: use target_options instead.
        target_protocol : str
            Used for loading the reference file, if it is a path. If None, protocol
            will be derived from the given path
        target_options : dict
            Extra FS options for loading the reference file ``fo``, if given as a path
        remote_protocol : str
            The protocol of the filesystem on which the references will be evaluated
            (unless fs is provided). If not given, will be derived from the first
            URL that has a protocol in the templates or in the references, in that
            order.
        remote_options : dict
            kwargs to go with remote_protocol
        fs : AbstractFileSystem | dict(str, (AbstractFileSystem | dict))
            Directly provide a file system(s):
                - a single filesystem instance
                - a dict of protocol:filesystem, where each value is either a filesystem
                  instance, or a dict of kwargs that can be used to create in
                  instance for the given protocol

            If this is given, remote_options and remote_protocol are ignored.
        template_overrides : dict
            Swap out any templates in the references file with these - useful for
            testing.
        simple_templates: bool
            Whether templates can be processed with simple replace (True) or if
            jinja  is needed (False, much slower). All reference sets produced by
            ``kerchunk`` are simple in this sense, but the spec allows for complex.
        max_gap, max_block: int
            For merging multiple concurrent requests to the same remote file.
            Neighboring byte ranges will only be merged when their
            inter-range gap is <= ``max_gap``. Default is 64KB. Set to 0
            to only merge when it requires no extra bytes. Pass a negative
            number to disable merging, appropriate for local target files.
            Neighboring byte ranges will only be merged when the size of
            the aggregated range is <= ``max_block``. Default is 256MB.
        cache_size : int
            Maximum size of LRU cache, where cache_size*record_size denotes
            the total number of references that can be loaded in memory at once.
            Only used for lazily loaded references.
        kwargs : passed to parent class
        """
        super().__init__(**kwargs)
        self.target = target
        self.template_overrides = template_overrides
        self.simple_templates = simple_templates
        self.templates = {}
        self.fss = {}
        self._dircache = {}
        self.max_gap = max_gap
        self.max_block = max_block
        if isinstance(fo, str):
            dic = dict(**ref_storage_args or target_options or {}, protocol=target_protocol)
            ref_fs, fo2 = fsspec.core.url_to_fs(fo, **dic)
            if ref_fs.isfile(fo2):
                with fsspec.open(fo, 'rb', **dic) as f:
                    logger.info('Read reference from URL %s', fo)
                    text = json.load(f)
                self._process_references(text, template_overrides)
            else:
                logger.info('Open lazy reference dict from URL %s', fo)
                self.references = LazyReferenceMapper(fo2, fs=ref_fs, cache_size=cache_size)
        else:
            self._process_references(fo, template_overrides)
        if isinstance(fs, dict):
            self.fss = {k: fsspec.filesystem(k.split(':', 1)[0], **opts) if isinstance(opts, dict) else opts for k, opts in fs.items()}
            if None not in self.fss:
                self.fss[None] = filesystem('file')
            return
        if fs is not None:
            remote_protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
            self.fss[remote_protocol] = fs
        if remote_protocol is None:
            for ref in self.templates.values():
                if callable(ref):
                    ref = ref()
                protocol, _ = fsspec.core.split_protocol(ref)
                if protocol and protocol not in self.fss:
                    fs = filesystem(protocol, **remote_options or {})
                    self.fss[protocol] = fs
        if remote_protocol is None:
            for ref in self.references.values():
                if callable(ref):
                    ref = ref()
                if isinstance(ref, list) and ref[0]:
                    protocol, _ = fsspec.core.split_protocol(ref[0])
                    if protocol not in self.fss:
                        fs = filesystem(protocol, **remote_options or {})
                        self.fss[protocol] = fs
                        break
        if remote_protocol and remote_protocol not in self.fss:
            fs = filesystem(remote_protocol, **remote_options or {})
            self.fss[remote_protocol] = fs
        self.fss[None] = fs or filesystem('file')

    def _cat_common(self, path, start=None, end=None):
        path = self._strip_protocol(path)
        logger.debug(f'cat: {path}')
        try:
            part = self.references[path]
        except KeyError:
            raise FileNotFoundError(path)
        if isinstance(part, str):
            part = part.encode()
        if isinstance(part, bytes):
            logger.debug(f'Reference: {path}, type bytes')
            if part.startswith(b'base64:'):
                part = base64.b64decode(part[7:])
            return (part, None, None)
        if len(part) == 1:
            logger.debug(f'Reference: {path}, whole file => {part}')
            url = part[0]
            start1, end1 = (start, end)
        else:
            url, start0, size = part
            logger.debug(f'Reference: {path} => {url}, offset {start0}, size {size}')
            end0 = start0 + size
            if start is not None:
                if start >= 0:
                    start1 = start0 + start
                else:
                    start1 = end0 + start
            else:
                start1 = start0
            if end is not None:
                if end >= 0:
                    end1 = start0 + end
                else:
                    end1 = end0 + end
            else:
                end1 = end0
        if url is None:
            url = self.target
        return (url, start1, end1)

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        part_or_url, start0, end0 = self._cat_common(path, start=start, end=end)
        if isinstance(part_or_url, bytes):
            return part_or_url[start:end]
        protocol, _ = split_protocol(part_or_url)
        try:
            await self.fss[protocol]._cat_file(part_or_url, start=start, end=end)
        except Exception as e:
            raise ReferenceNotReachable(path, part_or_url) from e

    def cat_file(self, path, start=None, end=None, **kwargs):
        part_or_url, start0, end0 = self._cat_common(path, start=start, end=end)
        if isinstance(part_or_url, bytes):
            return part_or_url[start:end]
        protocol, _ = split_protocol(part_or_url)
        try:
            return self.fss[protocol].cat_file(part_or_url, start=start0, end=end0)
        except Exception as e:
            raise ReferenceNotReachable(path, part_or_url) from e

    def pipe_file(self, path, value, **_):
        """Temporarily add binary data or reference as a file"""
        self.references[path] = value

    async def _get_file(self, rpath, lpath, **kwargs):
        if self.isdir(rpath):
            return os.makedirs(lpath, exist_ok=True)
        data = await self._cat_file(rpath)
        with open(lpath, 'wb') as f:
            f.write(data)

    def get_file(self, rpath, lpath, callback=DEFAULT_CALLBACK, **kwargs):
        if self.isdir(rpath):
            return os.makedirs(lpath, exist_ok=True)
        data = self.cat_file(rpath, **kwargs)
        callback.set_size(len(data))
        if isfilelike(lpath):
            lpath.write(data)
        else:
            with open(lpath, 'wb') as f:
                f.write(data)
        callback.absolute_update(len(data))

    def get(self, rpath, lpath, recursive=False, **kwargs):
        if recursive:
            self.ls('')
        rpath = self.expand_path(rpath, recursive=recursive)
        fs = fsspec.filesystem('file', auto_mkdir=True)
        targets = other_paths(rpath, lpath)
        if recursive:
            data = self.cat([r for r in rpath if not self.isdir(r)])
        else:
            data = self.cat(rpath)
        for remote, local in zip(rpath, targets):
            if remote in data:
                fs.pipe_file(local, data[remote])

    def cat(self, path, recursive=False, on_error='raise', **kwargs):
        if isinstance(path, str) and recursive:
            raise NotImplementedError
        if isinstance(path, list) and (recursive or any(('*' in p for p in path))):
            raise NotImplementedError
        proto_dict = _protocol_groups(path, self.references)
        out = {}
        for proto, paths in proto_dict.items():
            fs = self.fss[proto]
            urls, starts, ends, valid_paths = ([], [], [], [])
            for p in paths:
                try:
                    u, s, e = self._cat_common(p)
                except FileNotFoundError as err:
                    if on_error == 'raise':
                        raise
                    if on_error != 'omit':
                        out[p] = err
                else:
                    urls.append(u)
                    starts.append(s)
                    ends.append(e)
                    valid_paths.append(p)
            urls2 = []
            starts2 = []
            ends2 = []
            paths2 = []
            whole_files = set()
            for u, s, e, p in zip(urls, starts, ends, valid_paths):
                if isinstance(u, bytes):
                    out[p] = u
                elif s is None:
                    whole_files.add(u)
                    urls2.append(u)
                    starts2.append(s)
                    ends2.append(e)
                    paths2.append(p)
            for u, s, e, p in zip(urls, starts, ends, valid_paths):
                if s is not None and u not in whole_files:
                    urls2.append(u)
                    starts2.append(s)
                    ends2.append(e)
                    paths2.append(p)
            new_paths, new_starts, new_ends = merge_offset_ranges(list(urls2), list(starts2), list(ends2), sort=True, max_gap=self.max_gap, max_block=self.max_block)
            bytes_out = fs.cat_ranges(new_paths, new_starts, new_ends)
            for u, s, e, p in zip(urls, starts, ends, valid_paths):
                if p in out:
                    continue
                for np, ns, ne, b in zip(new_paths, new_starts, new_ends, bytes_out):
                    if np == u and (ns is None or ne is None):
                        if isinstance(b, Exception):
                            out[p] = b
                        else:
                            out[p] = b[s:e]
                    elif np == u and s >= ns and (e <= ne):
                        if isinstance(b, Exception):
                            out[p] = b
                        else:
                            out[p] = b[s - ns:e - ne or None]
        for k, v in out.copy().items():
            if isinstance(v, Exception) and k in self.references:
                ex = out[k]
                new_ex = ReferenceNotReachable(k, self.references[k])
                new_ex.__cause__ = ex
                if on_error == 'raise':
                    raise new_ex
                elif on_error != 'omit':
                    out[k] = new_ex
        if len(out) == 1 and isinstance(path, str) and ('*' not in path):
            return _first(out)
        return out

    def _process_references(self, references, template_overrides=None):
        vers = references.get('version', None)
        if vers is None:
            self._process_references0(references)
        elif vers == 1:
            self._process_references1(references, template_overrides=template_overrides)
        else:
            raise ValueError(f'Unknown reference spec version: {vers}')

    def _process_references0(self, references):
        """Make reference dict for Spec Version 0"""
        self.references = references

    def _process_references1(self, references, template_overrides=None):
        if not self.simple_templates or self.templates:
            import jinja2
        self.references = {}
        self._process_templates(references.get('templates', {}))

        @lru_cache(1000)
        def _render_jinja(u):
            return jinja2.Template(u).render(**self.templates)
        for k, v in references.get('refs', {}).items():
            if isinstance(v, str):
                if v.startswith('base64:'):
                    self.references[k] = base64.b64decode(v[7:])
                self.references[k] = v
            elif self.templates:
                u = v[0]
                if '{{' in u:
                    if self.simple_templates:
                        u = u.replace('{{', '{').replace('}}', '}').format(**self.templates)
                    else:
                        u = _render_jinja(u)
                self.references[k] = [u] if len(v) == 1 else [u, v[1], v[2]]
            else:
                self.references[k] = v
        self.references.update(self._process_gen(references.get('gen', [])))

    def _process_templates(self, tmp):
        self.templates = {}
        if self.template_overrides is not None:
            tmp.update(self.template_overrides)
        for k, v in tmp.items():
            if '{{' in v:
                import jinja2
                self.templates[k] = lambda temp=v, **kwargs: jinja2.Template(temp).render(**kwargs)
            else:
                self.templates[k] = v

    def _process_gen(self, gens):
        out = {}
        for gen in gens:
            dimension = {k: v if isinstance(v, list) else range(v.get('start', 0), v['stop'], v.get('step', 1)) for k, v in gen['dimensions'].items()}
            products = (dict(zip(dimension.keys(), values)) for values in itertools.product(*dimension.values()))
            for pr in products:
                import jinja2
                key = jinja2.Template(gen['key']).render(**pr, **self.templates)
                url = jinja2.Template(gen['url']).render(**pr, **self.templates)
                if 'offset' in gen and 'length' in gen:
                    offset = int(jinja2.Template(gen['offset']).render(**pr, **self.templates))
                    length = int(jinja2.Template(gen['length']).render(**pr, **self.templates))
                    out[key] = [url, offset, length]
                elif ('offset' in gen) ^ ('length' in gen):
                    raise ValueError("Both 'offset' and 'length' are required for a reference generator entry if either is provided.")
                else:
                    out[key] = [url]
        return out

    def _dircache_from_items(self):
        self.dircache = {'': []}
        it = self.references.items()
        for path, part in it:
            if isinstance(part, (bytes, str)):
                size = len(part)
            elif len(part) == 1:
                size = None
            else:
                _, _, size = part
            par = path.rsplit('/', 1)[0] if '/' in path else ''
            par0 = par
            subdirs = [par0]
            while par0 and par0 not in self.dircache:
                par0 = self._parent(par0)
                subdirs.append(par0)
            subdirs.reverse()
            for parent, child in zip(subdirs, subdirs[1:]):
                assert child not in self.dircache
                assert parent in self.dircache
                self.dircache[parent].append({'name': child, 'type': 'directory', 'size': 0})
                self.dircache[child] = []
            self.dircache[par].append({'name': path, 'type': 'file', 'size': size})

    def _open(self, path, mode='rb', block_size=None, cache_options=None, **kwargs):
        data = self.cat_file(path)
        return io.BytesIO(data)

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        if isinstance(self.references, LazyReferenceMapper):
            try:
                return self.references.ls(path, detail)
            except KeyError:
                pass
            raise FileNotFoundError(f"'{path}' is not a known key")
        if not self.dircache:
            self._dircache_from_items()
        out = self._ls_from_cache(path)
        if out is None:
            raise FileNotFoundError(path)
        if detail:
            return out
        return [o['name'] for o in out]

    def exists(self, path, **kwargs):
        return self.isdir(path) or self.isfile(path)

    def isdir(self, path):
        if self.dircache:
            return path in self.dircache
        elif isinstance(self.references, LazyReferenceMapper):
            return path in self.references.listdir('')
        else:
            return any((_.startswith(f'{path}/') for _ in self.references))

    def isfile(self, path):
        return path in self.references

    async def _ls(self, path, detail=True, **kwargs):
        return self.ls(path, detail, **kwargs)

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        if withdirs:
            return super().find(path, maxdepth=maxdepth, withdirs=withdirs, detail=detail, **kwargs)
        if path:
            path = self._strip_protocol(path)
            r = sorted((k for k in self.references if k.startswith(path)))
        else:
            r = sorted(self.references)
        if detail:
            if not self.dircache:
                self._dircache_from_items()
            return {k: self._ls_from_cache(k)[0] for k in r}
        else:
            return r

    def info(self, path, **kwargs):
        out = self.references.get(path)
        if out is not None:
            if isinstance(out, (str, bytes)):
                return {'name': path, 'type': 'file', 'size': len(out)}
            elif len(out) > 1:
                return {'name': path, 'type': 'file', 'size': out[2]}
            else:
                out0 = [{'name': path, 'type': 'file', 'size': None}]
        else:
            out = self.ls(path, True)
            out0 = [o for o in out if o['name'] == path]
            if not out0:
                return {'name': path, 'type': 'directory', 'size': 0}
        if out0[0]['size'] is None:
            prot, _ = split_protocol(self.references[path][0])
            out0[0]['size'] = self.fss[prot].size(self.references[path][0])
        return out0[0]

    async def _info(self, path, **kwargs):
        return self.info(path)

    async def _rm_file(self, path, **kwargs):
        self.references.pop(path, None)
        self.dircache.clear()

    async def _pipe_file(self, path, data):
        self.references[path] = data
        self.dircache.clear()

    async def _put_file(self, lpath, rpath, **kwargs):
        with open(lpath, 'rb') as f:
            self.references[rpath] = f.read()
        self.dircache.clear()

    def save_json(self, url, **storage_options):
        """Write modified references into new location"""
        out = {}
        for k, v in self.references.items():
            if isinstance(v, bytes):
                try:
                    out[k] = v.decode('ascii')
                except UnicodeDecodeError:
                    out[k] = (b'base64:' + base64.b64encode(v)).decode()
            else:
                out[k] = v
        with fsspec.open(url, 'wb', **storage_options) as f:
            f.write(json.dumps({'version': 1, 'refs': out}).encode())