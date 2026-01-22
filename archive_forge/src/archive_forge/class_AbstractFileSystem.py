from __future__ import annotations
import io
import logging
import os
import threading
import warnings
import weakref
from errno import ESPIPE
from glob import has_magic
from hashlib import sha256
from typing import ClassVar
from .callbacks import DEFAULT_CALLBACK
from .config import apply_config, conf
from .dircache import DirCache
from .transaction import Transaction
from .utils import (
class AbstractFileSystem(metaclass=_Cached):
    """
    An abstract super-class for pythonic file-systems

    Implementations are expected to be compatible with or, better, subclass
    from here.
    """
    cachable = True
    _cached = False
    blocksize = 2 ** 22
    sep = '/'
    protocol: ClassVar[str | tuple[str, ...]] = 'abstract'
    _latest = None
    async_impl = False
    mirror_sync_methods = False
    root_marker = ''
    transaction_type = Transaction
    _extra_tokenize_attributes = ()

    def __init__(self, *args, **storage_options):
        """Create and configure file-system instance

        Instances may be cachable, so if similar enough arguments are seen
        a new instance is not required. The token attribute exists to allow
        implementations to cache instances if they wish.

        A reasonable default should be provided if there are no arguments.

        Subclasses should call this method.

        Parameters
        ----------
        use_listings_cache, listings_expiry_time, max_paths:
            passed to ``DirCache``, if the implementation supports
            directory listing caching. Pass use_listings_cache=False
            to disable such caching.
        skip_instance_cache: bool
            If this is a cachable implementation, pass True here to force
            creating a new instance even if a matching instance exists, and prevent
            storing this instance.
        asynchronous: bool
        loop: asyncio-compatible IOLoop or None
        """
        if self._cached:
            return
        self._cached = True
        self._intrans = False
        self._transaction = None
        self._invalidated_caches_in_transaction = []
        self.dircache = DirCache(**storage_options)
        if storage_options.pop('add_docs', None):
            warnings.warn('add_docs is no longer supported.', FutureWarning)
        if storage_options.pop('add_aliases', None):
            warnings.warn('add_aliases has been removed.', FutureWarning)
        self._fs_token_ = None

    @property
    def fsid(self):
        """Persistent filesystem id that can be used to compare filesystems
        across sessions.
        """
        raise NotImplementedError

    @property
    def _fs_token(self):
        return self._fs_token_

    def __dask_tokenize__(self):
        return self._fs_token

    def __hash__(self):
        return int(self._fs_token, 16)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self._fs_token == other._fs_token

    def __reduce__(self):
        return (make_instance, (type(self), self.storage_args, self.storage_options))

    @classmethod
    def _strip_protocol(cls, path):
        """Turn path from fully-qualified to file-system-specific

        May require FS-specific handling, e.g., for relative paths or links.
        """
        if isinstance(path, list):
            return [cls._strip_protocol(p) for p in path]
        path = stringify_path(path)
        protos = (cls.protocol,) if isinstance(cls.protocol, str) else cls.protocol
        for protocol in protos:
            if path.startswith(protocol + '://'):
                path = path[len(protocol) + 3:]
            elif path.startswith(protocol + '::'):
                path = path[len(protocol) + 2:]
        path = path.rstrip('/')
        return path or cls.root_marker

    def unstrip_protocol(self, name: str) -> str:
        """Format FS-specific path to generic, including protocol"""
        protos = (self.protocol,) if isinstance(self.protocol, str) else self.protocol
        for protocol in protos:
            if name.startswith(f'{protocol}://'):
                return name
        return f'{protos[0]}://{name}'

    @staticmethod
    def _get_kwargs_from_urls(path):
        """If kwargs can be encoded in the paths, extract them here

        This should happen before instantiation of the class; incoming paths
        then should be amended to strip the options in methods.

        Examples may look like an sftp path "sftp://user@host:/my/path", where
        the user and host should become kwargs and later get stripped.
        """
        return {}

    @classmethod
    def current(cls):
        """Return the most recently instantiated FileSystem

        If no instance has been created, then create one with defaults
        """
        if cls._latest in cls._cache:
            return cls._cache[cls._latest]
        return cls()

    @property
    def transaction(self):
        """A context within which files are committed together upon exit

        Requires the file class to implement `.commit()` and `.discard()`
        for the normal and exception cases.
        """
        if self._transaction is None:
            self._transaction = self.transaction_type(self)
        return self._transaction

    def start_transaction(self):
        """Begin write transaction for deferring files, non-context version"""
        self._intrans = True
        self._transaction = self.transaction_type(self)
        return self.transaction

    def end_transaction(self):
        """Finish write transaction, non-context version"""
        self.transaction.complete()
        self._transaction = None
        for path in self._invalidated_caches_in_transaction:
            self.invalidate_cache(path)
        self._invalidated_caches_in_transaction.clear()

    def invalidate_cache(self, path=None):
        """
        Discard any cached directory information

        Parameters
        ----------
        path: string or None
            If None, clear all listings cached else listings at or under given
            path.
        """
        if self._intrans:
            self._invalidated_caches_in_transaction.append(path)

    def mkdir(self, path, create_parents=True, **kwargs):
        """
        Create directory entry at path

        For systems that don't have true directories, may create an for
        this instance only and not touch the real filesystem

        Parameters
        ----------
        path: str
            location
        create_parents: bool
            if True, this is equivalent to ``makedirs``
        kwargs:
            may be permissions, etc.
        """
        pass

    def makedirs(self, path, exist_ok=False):
        """Recursively make directories

        Creates directory at path and any intervening required directories.
        Raises exception if, for instance, the path already exists but is a
        file.

        Parameters
        ----------
        path: str
            leaf directory name
        exist_ok: bool (False)
            If False, will error if the target already exists
        """
        pass

    def rmdir(self, path):
        """Remove a directory, if empty"""
        pass

    def ls(self, path, detail=True, **kwargs):
        """List objects at path.

        This should include subdirectories and files at that location. The
        difference between a file and a directory must be clear when details
        are requested.

        The specific keys, or perhaps a FileInfo class, or similar, is TBD,
        but must be consistent across implementations.
        Must include:

        - full path to the entry (without protocol)
        - size of the entry, in bytes. If the value cannot be determined, will
          be ``None``.
        - type of entry, "file", "directory" or other

        Additional information
        may be present, appropriate to the file-system, e.g., generation,
        checksum, etc.

        May use refresh=True|False to allow use of self._ls_from_cache to
        check for a saved listing and avoid calling the backend. This would be
        common where listing may be expensive.

        Parameters
        ----------
        path: str
        detail: bool
            if True, gives a list of dictionaries, where each is the same as
            the result of ``info(path)``. If False, gives a list of paths
            (str).
        kwargs: may have additional backend-specific options, such as version
            information

        Returns
        -------
        List of strings if detail is False, or list of directory information
        dicts if detail is True.
        """
        raise NotImplementedError

    def _ls_from_cache(self, path):
        """Check cache for listing

        Returns listing, if found (may be empty list for a directly that exists
        but contains nothing), None if not in cache.
        """
        parent = self._parent(path)
        if path.rstrip('/') in self.dircache:
            return self.dircache[path.rstrip('/')]
        try:
            files = [f for f in self.dircache[parent] if f['name'] == path or (f['name'] == path.rstrip('/') and f['type'] == 'directory')]
            if len(files) == 0:
                raise FileNotFoundError(path)
            return files
        except KeyError:
            pass

    def walk(self, path, maxdepth=None, topdown=True, on_error='omit', **kwargs):
        """Return all files belows path

        List all files, recursing into subdirectories; output is iterator-style,
        like ``os.walk()``. For a simple list of files, ``find()`` is available.

        When topdown is True, the caller can modify the dirnames list in-place (perhaps
        using del or slice assignment), and walk() will
        only recurse into the subdirectories whose names remain in dirnames;
        this can be used to prune the search, impose a specific order of visiting,
        or even to inform walk() about directories the caller creates or renames before
        it resumes walk() again.
        Modifying dirnames when topdown is False has no effect. (see os.walk)

        Note that the "files" outputted will include anything that is not
        a directory, such as links.

        Parameters
        ----------
        path: str
            Root to recurse into
        maxdepth: int
            Maximum recursion depth. None means limitless, but not recommended
            on link-based file-systems.
        topdown: bool (True)
            Whether to walk the directory tree from the top downwards or from
            the bottom upwards.
        on_error: "omit", "raise", a collable
            if omit (default), path with exception will simply be empty;
            If raise, an underlying exception will be raised;
            if callable, it will be called with a single OSError instance as argument
        kwargs: passed to ``ls``
        """
        if maxdepth is not None and maxdepth < 1:
            raise ValueError('maxdepth must be at least 1')
        path = self._strip_protocol(path)
        full_dirs = {}
        dirs = {}
        files = {}
        detail = kwargs.pop('detail', False)
        try:
            listing = self.ls(path, detail=True, **kwargs)
        except (FileNotFoundError, OSError) as e:
            if on_error == 'raise':
                raise
            elif callable(on_error):
                on_error(e)
            if detail:
                return (path, {}, {})
            return (path, [], [])
        for info in listing:
            pathname = info['name'].rstrip('/')
            name = pathname.rsplit('/', 1)[-1]
            if info['type'] == 'directory' and pathname != path:
                full_dirs[name] = pathname
                dirs[name] = info
            elif pathname == path:
                files[''] = info
            else:
                files[name] = info
        if not detail:
            dirs = list(dirs)
            files = list(files)
        if topdown:
            yield (path, dirs, files)
        if maxdepth is not None:
            maxdepth -= 1
            if maxdepth < 1:
                if not topdown:
                    yield (path, dirs, files)
                return
        for d in dirs:
            yield from self.walk(full_dirs[d], maxdepth=maxdepth, detail=detail, topdown=topdown, **kwargs)
        if not topdown:
            yield (path, dirs, files)

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        """List all files below path.

        Like posix ``find`` command without conditions

        Parameters
        ----------
        path : str
        maxdepth: int or None
            If not None, the maximum number of levels to descend
        withdirs: bool
            Whether to include directory paths in the output. This is True
            when used by glob, but users usually only want files.
        kwargs are passed to ``ls``.
        """
        path = self._strip_protocol(path)
        out = {}
        if withdirs and path != '' and self.isdir(path):
            out[path] = self.info(path)
        for _, dirs, files in self.walk(path, maxdepth, detail=True, **kwargs):
            if withdirs:
                files.update(dirs)
            out.update({info['name']: info for name, info in files.items()})
        if not out and self.isfile(path):
            out[path] = {}
        names = sorted(out)
        if not detail:
            return names
        else:
            return {name: out[name] for name in names}

    def du(self, path, total=True, maxdepth=None, withdirs=False, **kwargs):
        """Space used by files and optionally directories within a path

        Directory size does not include the size of its contents.

        Parameters
        ----------
        path: str
        total: bool
            Whether to sum all the file sizes
        maxdepth: int or None
            Maximum number of directory levels to descend, None for unlimited.
        withdirs: bool
            Whether to include directory paths in the output.
        kwargs: passed to ``find``

        Returns
        -------
        Dict of {path: size} if total=False, or int otherwise, where numbers
        refer to bytes used.
        """
        sizes = {}
        if withdirs and self.isdir(path):
            info = self.info(path)
            sizes[info['name']] = info['size']
        for f in self.find(path, maxdepth=maxdepth, withdirs=withdirs, **kwargs):
            info = self.info(f)
            sizes[info['name']] = info['size']
        if total:
            return sum(sizes.values())
        else:
            return sizes

    def glob(self, path, maxdepth=None, **kwargs):
        """
        Find files by glob-matching.

        If the path ends with '/', only folders are returned.

        We support ``"**"``,
        ``"?"`` and ``"[..]"``. We do not support ^ for pattern negation.

        The `maxdepth` option is applied on the first `**` found in the path.

        kwargs are passed to ``ls``.
        """
        if maxdepth is not None and maxdepth < 1:
            raise ValueError('maxdepth must be at least 1')
        import re
        seps = (os.path.sep, os.path.altsep) if os.path.altsep else (os.path.sep,)
        ends_with_sep = path.endswith(seps)
        path = self._strip_protocol(path)
        append_slash_to_dirname = ends_with_sep or path.endswith(tuple((sep + '**' for sep in seps)))
        idx_star = path.find('*') if path.find('*') >= 0 else len(path)
        idx_qmark = path.find('?') if path.find('?') >= 0 else len(path)
        idx_brace = path.find('[') if path.find('[') >= 0 else len(path)
        min_idx = min(idx_star, idx_qmark, idx_brace)
        detail = kwargs.pop('detail', False)
        if not has_magic(path):
            if self.exists(path, **kwargs):
                if not detail:
                    return [path]
                else:
                    return {path: self.info(path, **kwargs)}
            elif not detail:
                return []
            else:
                return {}
        elif '/' in path[:min_idx]:
            min_idx = path[:min_idx].rindex('/')
            root = path[:min_idx + 1]
            depth = path[min_idx + 1:].count('/') + 1
        else:
            root = ''
            depth = path[min_idx + 1:].count('/') + 1
        if '**' in path:
            if maxdepth is not None:
                idx_double_stars = path.find('**')
                depth_double_stars = path[idx_double_stars:].count('/') + 1
                depth = depth - depth_double_stars + maxdepth
            else:
                depth = None
        allpaths = self.find(root, maxdepth=depth, withdirs=True, detail=True, **kwargs)
        pattern = glob_translate(path + ('/' if ends_with_sep else ''))
        pattern = re.compile(pattern)
        out = {p: info for p, info in sorted(allpaths.items()) if pattern.match(p + '/' if append_slash_to_dirname and info['type'] == 'directory' else p)}
        if detail:
            return out
        else:
            return list(out)

    def exists(self, path, **kwargs):
        """Is there a file at the given path"""
        try:
            self.info(path, **kwargs)
            return True
        except:
            return False

    def lexists(self, path, **kwargs):
        """If there is a file at the given path (including
        broken links)"""
        return self.exists(path)

    def info(self, path, **kwargs):
        """Give details of entry at path

        Returns a single dictionary, with exactly the same information as ``ls``
        would with ``detail=True``.

        The default implementation should calls ls and could be overridden by a
        shortcut. kwargs are passed on to ```ls()``.

        Some file systems might not be able to measure the file's size, in
        which case, the returned dict will include ``'size': None``.

        Returns
        -------
        dict with keys: name (full path in the FS), size (in bytes), type (file,
        directory, or something else) and other FS-specific keys.
        """
        path = self._strip_protocol(path)
        out = self.ls(self._parent(path), detail=True, **kwargs)
        out = [o for o in out if o['name'].rstrip('/') == path]
        if out:
            return out[0]
        out = self.ls(path, detail=True, **kwargs)
        path = path.rstrip('/')
        out1 = [o for o in out if o['name'].rstrip('/') == path]
        if len(out1) == 1:
            if 'size' not in out1[0]:
                out1[0]['size'] = None
            return out1[0]
        elif len(out1) > 1 or out:
            return {'name': path, 'size': 0, 'type': 'directory'}
        else:
            raise FileNotFoundError(path)

    def checksum(self, path):
        """Unique value for current version of file

        If the checksum is the same from one moment to another, the contents
        are guaranteed to be the same. If the checksum changes, the contents
        *might* have changed.

        This should normally be overridden; default will probably capture
        creation/modification timestamp (which would be good) or maybe
        access timestamp (which would be bad)
        """
        return int(tokenize(self.info(path)), 16)

    def size(self, path):
        """Size in bytes of file"""
        return self.info(path).get('size', None)

    def sizes(self, paths):
        """Size in bytes of each file in a list of paths"""
        return [self.size(p) for p in paths]

    def isdir(self, path):
        """Is this entry directory-like?"""
        try:
            return self.info(path)['type'] == 'directory'
        except OSError:
            return False

    def isfile(self, path):
        """Is this entry file-like?"""
        try:
            return self.info(path)['type'] == 'file'
        except:
            return False

    def read_text(self, path, encoding=None, errors=None, newline=None, **kwargs):
        """Get the contents of the file as a string.

        Parameters
        ----------
        path: str
            URL of file on this filesystems
        encoding, errors, newline: same as `open`.
        """
        with self.open(path, mode='r', encoding=encoding, errors=errors, newline=newline, **kwargs) as f:
            return f.read()

    def write_text(self, path, value, encoding=None, errors=None, newline=None, **kwargs):
        """Write the text to the given file.

        An existing file will be overwritten.

        Parameters
        ----------
        path: str
            URL of file on this filesystems
        value: str
            Text to write.
        encoding, errors, newline: same as `open`.
        """
        with self.open(path, mode='w', encoding=encoding, errors=errors, newline=newline, **kwargs) as f:
            return f.write(value)

    def cat_file(self, path, start=None, end=None, **kwargs):
        """Get the content of a file

        Parameters
        ----------
        path: URL of file on this filesystems
        start, end: int
            Bytes limits of the read. If negative, backwards from end,
            like usual python slices. Either can be None for start or
            end of file, respectively
        kwargs: passed to ``open()``.
        """
        with self.open(path, 'rb', **kwargs) as f:
            if start is not None:
                if start >= 0:
                    f.seek(start)
                else:
                    f.seek(max(0, f.size + start))
            if end is not None:
                if end < 0:
                    end = f.size + end
                return f.read(end - f.tell())
            return f.read()

    def pipe_file(self, path, value, **kwargs):
        """Set the bytes of given file"""
        with self.open(path, 'wb', **kwargs) as f:
            f.write(value)

    def pipe(self, path, value=None, **kwargs):
        """Put value into path

        (counterpart to ``cat``)

        Parameters
        ----------
        path: string or dict(str, bytes)
            If a string, a single remote location to put ``value`` bytes; if a dict,
            a mapping of {path: bytesvalue}.
        value: bytes, optional
            If using a single path, these are the bytes to put there. Ignored if
            ``path`` is a dict
        """
        if isinstance(path, str):
            self.pipe_file(self._strip_protocol(path), value, **kwargs)
        elif isinstance(path, dict):
            for k, v in path.items():
                self.pipe_file(self._strip_protocol(k), v, **kwargs)
        else:
            raise ValueError('path must be str or dict')

    def cat_ranges(self, paths, starts, ends, max_gap=None, on_error='return', **kwargs):
        """Get the contents of byte ranges from one or more files

        Parameters
        ----------
        paths: list
            A list of of filepaths on this filesystems
        starts, ends: int or list
            Bytes limits of the read. If using a single int, the same value will be
            used to read all the specified files.
        """
        if max_gap is not None:
            raise NotImplementedError
        if not isinstance(paths, list):
            raise TypeError
        if not isinstance(starts, list):
            starts = [starts] * len(paths)
        if not isinstance(ends, list):
            ends = [ends] * len(paths)
        if len(starts) != len(paths) or len(ends) != len(paths):
            raise ValueError
        out = []
        for p, s, e in zip(paths, starts, ends):
            try:
                out.append(self.cat_file(p, s, e))
            except Exception as e:
                if on_error == 'return':
                    out.append(e)
                else:
                    raise
        return out

    def cat(self, path, recursive=False, on_error='raise', **kwargs):
        """Fetch (potentially multiple) paths' contents

        Parameters
        ----------
        recursive: bool
            If True, assume the path(s) are directories, and get all the
            contained files
        on_error : "raise", "omit", "return"
            If raise, an underlying exception will be raised (converted to KeyError
            if the type is in self.missing_exceptions); if omit, keys with exception
            will simply not be included in the output; if "return", all keys are
            included in the output, but the value will be bytes or an exception
            instance.
        kwargs: passed to cat_file

        Returns
        -------
        dict of {path: contents} if there are multiple paths
        or the path has been otherwise expanded
        """
        paths = self.expand_path(path, recursive=recursive)
        if len(paths) > 1 or isinstance(path, list) or paths[0] != self._strip_protocol(path):
            out = {}
            for path in paths:
                try:
                    out[path] = self.cat_file(path, **kwargs)
                except Exception as e:
                    if on_error == 'raise':
                        raise
                    if on_error == 'return':
                        out[path] = e
            return out
        else:
            return self.cat_file(paths[0], **kwargs)

    def get_file(self, rpath, lpath, callback=DEFAULT_CALLBACK, outfile=None, **kwargs):
        """Copy single remote file to local"""
        from .implementations.local import LocalFileSystem
        if isfilelike(lpath):
            outfile = lpath
        elif self.isdir(rpath):
            os.makedirs(lpath, exist_ok=True)
            return None
        fs = LocalFileSystem(auto_mkdir=True)
        fs.makedirs(fs._parent(lpath), exist_ok=True)
        with self.open(rpath, 'rb', **kwargs) as f1:
            if outfile is None:
                outfile = open(lpath, 'wb')
            try:
                callback.set_size(getattr(f1, 'size', None))
                data = True
                while data:
                    data = f1.read(self.blocksize)
                    segment_len = outfile.write(data)
                    if segment_len is None:
                        segment_len = len(data)
                    callback.relative_update(segment_len)
            finally:
                if not isfilelike(lpath):
                    outfile.close()

    def get(self, rpath, lpath, recursive=False, callback=DEFAULT_CALLBACK, maxdepth=None, **kwargs):
        """Copy file(s) to local.

        Copies a specific file or tree of files (if recursive=True). If lpath
        ends with a "/", it will be assumed to be a directory, and target files
        will go within. Can submit a list of paths, which may be glob-patterns
        and will be expanded.

        Calls get_file for each source.
        """
        if isinstance(lpath, list) and isinstance(rpath, list):
            rpaths = rpath
            lpaths = lpath
        else:
            from .implementations.local import LocalFileSystem, make_path_posix, trailing_sep
            source_is_str = isinstance(rpath, str)
            rpaths = self.expand_path(rpath, recursive=recursive, maxdepth=maxdepth)
            if source_is_str and (not recursive or maxdepth is not None):
                rpaths = [p for p in rpaths if not (trailing_sep(p) or self.isdir(p))]
                if not rpaths:
                    return
            if isinstance(lpath, str):
                lpath = make_path_posix(lpath)
            source_is_file = len(rpaths) == 1
            dest_is_dir = isinstance(lpath, str) and (trailing_sep(lpath) or LocalFileSystem().isdir(lpath))
            exists = source_is_str and (has_magic(rpath) and source_is_file or (not has_magic(rpath) and dest_is_dir and (not trailing_sep(rpath))))
            lpaths = other_paths(rpaths, lpath, exists=exists, flatten=not source_is_str)
        callback.set_size(len(lpaths))
        for lpath, rpath in callback.wrap(zip(lpaths, rpaths)):
            with callback.branched(rpath, lpath) as child:
                self.get_file(rpath, lpath, callback=child, **kwargs)

    def put_file(self, lpath, rpath, callback=DEFAULT_CALLBACK, **kwargs):
        """Copy single file to remote"""
        if os.path.isdir(lpath):
            self.makedirs(rpath, exist_ok=True)
            return None
        with open(lpath, 'rb') as f1:
            size = f1.seek(0, 2)
            callback.set_size(size)
            f1.seek(0)
            self.mkdirs(self._parent(os.fspath(rpath)), exist_ok=True)
            with self.open(rpath, 'wb', **kwargs) as f2:
                while f1.tell() < size:
                    data = f1.read(self.blocksize)
                    segment_len = f2.write(data)
                    if segment_len is None:
                        segment_len = len(data)
                    callback.relative_update(segment_len)

    def put(self, lpath, rpath, recursive=False, callback=DEFAULT_CALLBACK, maxdepth=None, **kwargs):
        """Copy file(s) from local.

        Copies a specific file or tree of files (if recursive=True). If rpath
        ends with a "/", it will be assumed to be a directory, and target files
        will go within.

        Calls put_file for each source.
        """
        if isinstance(lpath, list) and isinstance(rpath, list):
            rpaths = rpath
            lpaths = lpath
        else:
            from .implementations.local import LocalFileSystem, make_path_posix, trailing_sep
            source_is_str = isinstance(lpath, str)
            if source_is_str:
                lpath = make_path_posix(lpath)
            fs = LocalFileSystem()
            lpaths = fs.expand_path(lpath, recursive=recursive, maxdepth=maxdepth)
            if source_is_str and (not recursive or maxdepth is not None):
                lpaths = [p for p in lpaths if not (trailing_sep(p) or fs.isdir(p))]
                if not lpaths:
                    return
            source_is_file = len(lpaths) == 1
            dest_is_dir = isinstance(rpath, str) and (trailing_sep(rpath) or self.isdir(rpath))
            rpath = self._strip_protocol(rpath) if isinstance(rpath, str) else [self._strip_protocol(p) for p in rpath]
            exists = source_is_str and (has_magic(lpath) and source_is_file or (not has_magic(lpath) and dest_is_dir and (not trailing_sep(lpath))))
            rpaths = other_paths(lpaths, rpath, exists=exists, flatten=not source_is_str)
        callback.set_size(len(rpaths))
        for lpath, rpath in callback.wrap(zip(lpaths, rpaths)):
            with callback.branched(lpath, rpath) as child:
                self.put_file(lpath, rpath, callback=child, **kwargs)

    def head(self, path, size=1024):
        """Get the first ``size`` bytes from file"""
        with self.open(path, 'rb') as f:
            return f.read(size)

    def tail(self, path, size=1024):
        """Get the last ``size`` bytes from file"""
        with self.open(path, 'rb') as f:
            f.seek(max(-size, -f.size), 2)
            return f.read()

    def cp_file(self, path1, path2, **kwargs):
        raise NotImplementedError

    def copy(self, path1, path2, recursive=False, maxdepth=None, on_error=None, **kwargs):
        """Copy within two locations in the filesystem

        on_error : "raise", "ignore"
            If raise, any not-found exceptions will be raised; if ignore any
            not-found exceptions will cause the path to be skipped; defaults to
            raise unless recursive is true, where the default is ignore
        """
        if on_error is None and recursive:
            on_error = 'ignore'
        elif on_error is None:
            on_error = 'raise'
        if isinstance(path1, list) and isinstance(path2, list):
            paths1 = path1
            paths2 = path2
        else:
            from .implementations.local import trailing_sep
            source_is_str = isinstance(path1, str)
            paths1 = self.expand_path(path1, recursive=recursive, maxdepth=maxdepth)
            if source_is_str and (not recursive or maxdepth is not None):
                paths1 = [p for p in paths1 if not (trailing_sep(p) or self.isdir(p))]
                if not paths1:
                    return
            source_is_file = len(paths1) == 1
            dest_is_dir = isinstance(path2, str) and (trailing_sep(path2) or self.isdir(path2))
            exists = source_is_str and (has_magic(path1) and source_is_file or (not has_magic(path1) and dest_is_dir and (not trailing_sep(path1))))
            paths2 = other_paths(paths1, path2, exists=exists, flatten=not source_is_str)
        for p1, p2 in zip(paths1, paths2):
            try:
                self.cp_file(p1, p2, **kwargs)
            except FileNotFoundError:
                if on_error == 'raise':
                    raise

    def expand_path(self, path, recursive=False, maxdepth=None, **kwargs):
        """Turn one or more globs or directories into a list of all matching paths
        to files or directories.

        kwargs are passed to ``glob`` or ``find``, which may in turn call ``ls``
        """
        if maxdepth is not None and maxdepth < 1:
            raise ValueError('maxdepth must be at least 1')
        if isinstance(path, (str, os.PathLike)):
            out = self.expand_path([path], recursive, maxdepth)
        else:
            out = set()
            path = [self._strip_protocol(p) for p in path]
            for p in path:
                if has_magic(p):
                    bit = set(self.glob(p, maxdepth=maxdepth, **kwargs))
                    out |= bit
                    if recursive:
                        if maxdepth is not None and maxdepth <= 1:
                            continue
                        out |= set(self.expand_path(list(bit), recursive=recursive, maxdepth=maxdepth - 1 if maxdepth is not None else None, **kwargs))
                    continue
                elif recursive:
                    rec = set(self.find(p, maxdepth=maxdepth, withdirs=True, detail=False, **kwargs))
                    out |= rec
                if p not in out and (recursive is False or self.exists(p)):
                    out.add(p)
        if not out:
            raise FileNotFoundError(path)
        return sorted(out)

    def mv(self, path1, path2, recursive=False, maxdepth=None, **kwargs):
        """Move file(s) from one location to another"""
        if path1 == path2:
            logger.debug('%s mv: The paths are the same, so no files were moved.', self)
        else:
            self.copy(path1, path2, recursive=recursive, maxdepth=maxdepth)
            self.rm(path1, recursive=recursive)

    def rm_file(self, path):
        """Delete a file"""
        self._rm(path)

    def _rm(self, path):
        """Delete one file"""
        raise NotImplementedError

    def rm(self, path, recursive=False, maxdepth=None):
        """Delete files.

        Parameters
        ----------
        path: str or list of str
            File(s) to delete.
        recursive: bool
            If file(s) are directories, recursively delete contents and then
            also remove the directory
        maxdepth: int or None
            Depth to pass to walk for finding files to delete, if recursive.
            If None, there will be no limit and infinite recursion may be
            possible.
        """
        path = self.expand_path(path, recursive=recursive, maxdepth=maxdepth)
        for p in reversed(path):
            self.rm_file(p)

    @classmethod
    def _parent(cls, path):
        path = cls._strip_protocol(path)
        if '/' in path:
            parent = path.rsplit('/', 1)[0].lstrip(cls.root_marker)
            return cls.root_marker + parent
        else:
            return cls.root_marker

    def _open(self, path, mode='rb', block_size=None, autocommit=True, cache_options=None, **kwargs):
        """Return raw bytes-mode file-like from the file-system"""
        return AbstractBufferedFile(self, path, mode, block_size, autocommit, cache_options=cache_options, **kwargs)

    def open(self, path, mode='rb', block_size=None, cache_options=None, compression=None, **kwargs):
        """
        Return a file-like object from the filesystem

        The resultant instance must function correctly in a context ``with``
        block.

        Parameters
        ----------
        path: str
            Target file
        mode: str like 'rb', 'w'
            See builtin ``open()``
        block_size: int
            Some indication of buffering - this is a value in bytes
        cache_options : dict, optional
            Extra arguments to pass through to the cache.
        compression: string or None
            If given, open file using compression codec. Can either be a compression
            name (a key in ``fsspec.compression.compr``) or "infer" to guess the
            compression from the filename suffix.
        encoding, errors, newline: passed on to TextIOWrapper for text mode
        """
        import io
        path = self._strip_protocol(path)
        if 'b' not in mode:
            mode = mode.replace('t', '') + 'b'
            text_kwargs = {k: kwargs.pop(k) for k in ['encoding', 'errors', 'newline'] if k in kwargs}
            return io.TextIOWrapper(self.open(path, mode, block_size=block_size, cache_options=cache_options, compression=compression, **kwargs), **text_kwargs)
        else:
            ac = kwargs.pop('autocommit', not self._intrans)
            f = self._open(path, mode=mode, block_size=block_size, autocommit=ac, cache_options=cache_options, **kwargs)
            if compression is not None:
                from fsspec.compression import compr
                from fsspec.core import get_compression
                compression = get_compression(path, compression)
                compress = compr[compression]
                f = compress(f, mode=mode[0])
            if not ac and 'r' not in mode:
                self.transaction.files.append(f)
            return f

    def touch(self, path, truncate=True, **kwargs):
        """Create empty file, or update timestamp

        Parameters
        ----------
        path: str
            file location
        truncate: bool
            If True, always set file size to 0; if False, update timestamp and
            leave file unchanged, if backend allows this
        """
        if truncate or not self.exists(path):
            with self.open(path, 'wb', **kwargs):
                pass
        else:
            raise NotImplementedError

    def ukey(self, path):
        """Hash of file properties, to tell if it has changed"""
        return sha256(str(self.info(path)).encode()).hexdigest()

    def read_block(self, fn, offset, length, delimiter=None):
        """Read a block of bytes from

        Starting at ``offset`` of the file, read ``length`` bytes.  If
        ``delimiter`` is set then we ensure that the read starts and stops at
        delimiter boundaries that follow the locations ``offset`` and ``offset
        + length``.  If ``offset`` is zero then we start at zero.  The
        bytestring returned WILL include the end delimiter string.

        If offset+length is beyond the eof, reads to eof.

        Parameters
        ----------
        fn: string
            Path to filename
        offset: int
            Byte offset to start read
        length: int
            Number of bytes to read. If None, read to end.
        delimiter: bytes (optional)
            Ensure reading starts and stops at delimiter bytestring

        Examples
        --------
        >>> fs.read_block('data/file.csv', 0, 13)  # doctest: +SKIP
        b'Alice, 100\\nBo'
        >>> fs.read_block('data/file.csv', 0, 13, delimiter=b'\\n')  # doctest: +SKIP
        b'Alice, 100\\nBob, 200\\n'

        Use ``length=None`` to read to the end of the file.
        >>> fs.read_block('data/file.csv', 0, None, delimiter=b'\\n')  # doctest: +SKIP
        b'Alice, 100\\nBob, 200\\nCharlie, 300'

        See Also
        --------
        :func:`fsspec.utils.read_block`
        """
        with self.open(fn, 'rb') as f:
            size = f.size
            if length is None:
                length = size
            if size is not None and offset + length > size:
                length = size - offset
            return read_block(f, offset, length, delimiter)

    def to_json(self):
        """
        JSON representation of this filesystem instance

        Returns
        -------
        str: JSON structure with keys cls (the python location of this class),
            protocol (text name of this class's protocol, first one in case of
            multiple), args (positional args, usually empty), and all other
            kwargs as their own keys.
        """
        import json
        cls = type(self)
        cls = '.'.join((cls.__module__, cls.__name__))
        proto = self.protocol[0] if isinstance(self.protocol, (tuple, list)) else self.protocol
        return json.dumps(dict(cls=cls, protocol=proto, args=self.storage_args, **self.storage_options))

    @staticmethod
    def from_json(blob):
        """
        Recreate a filesystem instance from JSON representation

        See ``.to_json()`` for the expected structure of the input

        Parameters
        ----------
        blob: str

        Returns
        -------
        file system instance, not necessarily of this particular class.
        """
        import json
        from .registry import _import_class, get_filesystem_class
        dic = json.loads(blob)
        protocol = dic.pop('protocol')
        try:
            cls = _import_class(dic.pop('cls'))
        except (ImportError, ValueError, RuntimeError, KeyError):
            cls = get_filesystem_class(protocol)
        return cls(*dic.pop('args', ()), **dic)

    def _get_pyarrow_filesystem(self):
        """
        Make a version of the FS instance which will be acceptable to pyarrow
        """
        return self

    def get_mapper(self, root='', check=False, create=False, missing_exceptions=None):
        """Create key/value store based on this file-system

        Makes a MutableMapping interface to the FS at the given root path.
        See ``fsspec.mapping.FSMap`` for further details.
        """
        from .mapping import FSMap
        return FSMap(root, self, check=check, create=create, missing_exceptions=missing_exceptions)

    @classmethod
    def clear_instance_cache(cls):
        """
        Clear the cache of filesystem instances.

        Notes
        -----
        Unless overridden by setting the ``cachable`` class attribute to False,
        the filesystem class stores a reference to newly created instances. This
        prevents Python's normal rules around garbage collection from working,
        since the instances refcount will not drop to zero until
        ``clear_instance_cache`` is called.
        """
        cls._cache.clear()

    def created(self, path):
        """Return the created timestamp of a file as a datetime.datetime"""
        raise NotImplementedError

    def modified(self, path):
        """Return the modified timestamp of a file as a datetime.datetime"""
        raise NotImplementedError

    def read_bytes(self, path, start=None, end=None, **kwargs):
        """Alias of `AbstractFileSystem.cat_file`."""
        return self.cat_file(path, start=start, end=end, **kwargs)

    def write_bytes(self, path, value, **kwargs):
        """Alias of `AbstractFileSystem.pipe_file`."""
        self.pipe_file(path, value, **kwargs)

    def makedir(self, path, create_parents=True, **kwargs):
        """Alias of `AbstractFileSystem.mkdir`."""
        return self.mkdir(path, create_parents=create_parents, **kwargs)

    def mkdirs(self, path, exist_ok=False):
        """Alias of `AbstractFileSystem.makedirs`."""
        return self.makedirs(path, exist_ok=exist_ok)

    def listdir(self, path, detail=True, **kwargs):
        """Alias of `AbstractFileSystem.ls`."""
        return self.ls(path, detail=detail, **kwargs)

    def cp(self, path1, path2, **kwargs):
        """Alias of `AbstractFileSystem.copy`."""
        return self.copy(path1, path2, **kwargs)

    def move(self, path1, path2, **kwargs):
        """Alias of `AbstractFileSystem.mv`."""
        return self.mv(path1, path2, **kwargs)

    def stat(self, path, **kwargs):
        """Alias of `AbstractFileSystem.info`."""
        return self.info(path, **kwargs)

    def disk_usage(self, path, total=True, maxdepth=None, **kwargs):
        """Alias of `AbstractFileSystem.du`."""
        return self.du(path, total=total, maxdepth=maxdepth, **kwargs)

    def rename(self, path1, path2, **kwargs):
        """Alias of `AbstractFileSystem.mv`."""
        return self.mv(path1, path2, **kwargs)

    def delete(self, path, recursive=False, maxdepth=None):
        """Alias of `AbstractFileSystem.rm`."""
        return self.rm(path, recursive=recursive, maxdepth=maxdepth)

    def upload(self, lpath, rpath, recursive=False, **kwargs):
        """Alias of `AbstractFileSystem.put`."""
        return self.put(lpath, rpath, recursive=recursive, **kwargs)

    def download(self, rpath, lpath, recursive=False, **kwargs):
        """Alias of `AbstractFileSystem.get`."""
        return self.get(rpath, lpath, recursive=recursive, **kwargs)

    def sign(self, path, expiration=100, **kwargs):
        """Create a signed URL representing the given path

        Some implementations allow temporary URLs to be generated, as a
        way of delegating credentials.

        Parameters
        ----------
        path : str
             The path on the filesystem
        expiration : int
            Number of seconds to enable the URL for (if supported)

        Returns
        -------
        URL : str
            The signed URL

        Raises
        ------
        NotImplementedError : if method is not implemented for a filesystem
        """
        raise NotImplementedError('Sign is not implemented for this filesystem')

    def _isfilestore(self):
        return False