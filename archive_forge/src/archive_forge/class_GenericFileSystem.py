from __future__ import annotations
import inspect
import logging
import os
import shutil
import uuid
from typing import Optional
from .asyn import AsyncFileSystem, _run_coros_in_chunks, sync_wrapper
from .callbacks import DEFAULT_CALLBACK
from .core import filesystem, get_filesystem_class, split_protocol, url_to_fs
class GenericFileSystem(AsyncFileSystem):
    """Wrapper over all other FS types

    <experimental!>

    This implementation is a single unified interface to be able to run FS operations
    over generic URLs, and dispatch to the specific implementations using the URL
    protocol prefix.

    Note: instances of this FS are always async, even if you never use it with any async
    backend.
    """
    protocol = 'generic'

    def __init__(self, default_method='default', **kwargs):
        """

        Parameters
        ----------
        default_method: str (optional)
            Defines how to configure backend FS instances. Options are:
            - "default": instantiate like FSClass(), with no
              extra arguments; this is the default instance of that FS, and can be
              configured via the config system
            - "generic": takes instances from the `_generic_fs` dict in this module,
              which you must populate before use. Keys are by protocol
            - "current": takes the most recently instantiated version of each FS
        """
        self.method = default_method
        super().__init__(**kwargs)

    def _parent(self, path):
        fs = _resolve_fs(path, self.method)
        return fs.unstrip_protocol(fs._parent(path))

    def _strip_protocol(self, path):
        fs = _resolve_fs(path, self.method)
        return fs.unstrip_protocol(fs._strip_protocol(path))

    async def _find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        fs = _resolve_fs(path, self.method)
        if fs.async_impl:
            out = await fs._find(path, maxdepth=maxdepth, withdirs=withdirs, detail=True, **kwargs)
        else:
            out = fs.find(path, maxdepth=maxdepth, withdirs=withdirs, detail=True, **kwargs)
        result = {}
        for k, v in out.items():
            name = fs.unstrip_protocol(k)
            v['name'] = name
            result[name] = v
        if detail:
            return result
        return list(result)

    async def _info(self, url, **kwargs):
        fs = _resolve_fs(url, self.method)
        if fs.async_impl:
            out = await fs._info(url, **kwargs)
        else:
            out = fs.info(url, **kwargs)
        out['name'] = fs.unstrip_protocol(out['name'])
        return out

    async def _ls(self, url, detail=True, **kwargs):
        fs = _resolve_fs(url, self.method)
        if fs.async_impl:
            out = await fs._ls(url, detail=True, **kwargs)
        else:
            out = fs.ls(url, detail=True, **kwargs)
        for o in out:
            o['name'] = fs.unstrip_protocol(o['name'])
        if detail:
            return out
        else:
            return [o['name'] for o in out]

    async def _cat_file(self, url, **kwargs):
        fs = _resolve_fs(url, self.method)
        if fs.async_impl:
            return await fs._cat_file(url, **kwargs)
        else:
            return fs.cat_file(url, **kwargs)

    async def _pipe_file(self, path, value, **kwargs):
        fs = _resolve_fs(path, self.method)
        if fs.async_impl:
            return await fs._pipe_file(path, value, **kwargs)
        else:
            return fs.pipe_file(path, value, **kwargs)

    async def _rm(self, url, **kwargs):
        urls = url
        if isinstance(urls, str):
            urls = [urls]
        fs = _resolve_fs(urls[0], self.method)
        if fs.async_impl:
            await fs._rm(urls, **kwargs)
        else:
            fs.rm(url, **kwargs)

    async def _makedirs(self, path, exist_ok=False):
        logger.debug('Make dir %s', path)
        fs = _resolve_fs(path, self.method)
        if fs.async_impl:
            await fs._makedirs(path, exist_ok=exist_ok)
        else:
            fs.makedirs(path, exist_ok=exist_ok)

    def rsync(self, source, destination, **kwargs):
        """Sync files between two directory trees

        See `func:rsync` for more details.
        """
        rsync(source, destination, fs=self, **kwargs)

    async def _cp_file(self, url, url2, blocksize=2 ** 20, callback=DEFAULT_CALLBACK, **kwargs):
        fs = _resolve_fs(url, self.method)
        fs2 = _resolve_fs(url2, self.method)
        if fs is fs2:
            if fs.async_impl:
                return await fs._cp_file(url, url2, **kwargs)
            else:
                return fs.cp_file(url, url2, **kwargs)
        kw = {'blocksize': 0, 'cache_type': 'none'}
        try:
            f1 = await fs.open_async(url, 'rb') if hasattr(fs, 'open_async') else fs.open(url, 'rb', **kw)
            callback.set_size(await maybe_await(f1.size))
            f2 = await fs2.open_async(url2, 'wb') if hasattr(fs2, 'open_async') else fs2.open(url2, 'wb', **kw)
            while f1.size is None or f2.tell() < f1.size:
                data = await maybe_await(f1.read(blocksize))
                if f1.size is None and (not data):
                    break
                await maybe_await(f2.write(data))
                callback.absolute_update(f2.tell())
        finally:
            try:
                await maybe_await(f2.close())
                await maybe_await(f1.close())
            except NameError:
                pass

    async def _make_many_dirs(self, urls, exist_ok=True):
        fs = _resolve_fs(urls[0], self.method)
        if fs.async_impl:
            coros = [fs._makedirs(u, exist_ok=exist_ok) for u in urls]
            await _run_coros_in_chunks(coros)
        else:
            for u in urls:
                fs.makedirs(u, exist_ok=exist_ok)
    make_many_dirs = sync_wrapper(_make_many_dirs)

    async def _copy(self, path1: list[str], path2: list[str], recursive: bool=False, on_error: str='ignore', maxdepth: Optional[int]=None, batch_size: Optional[int]=None, tempdir: Optional[str]=None, **kwargs):
        if recursive:
            raise NotImplementedError
        fs = _resolve_fs(path1[0], self.method)
        fs2 = _resolve_fs(path2[0], self.method)
        if fs is fs2:
            if fs.async_impl:
                return await fs._copy(path1, path2, **kwargs)
            else:
                return fs.copy(path1, path2, **kwargs)
        await copy_file_op(fs, path1, fs2, path2, tempdir, batch_size, on_error=on_error)