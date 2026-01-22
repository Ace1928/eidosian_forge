import asyncio
import io
import logging
import re
import weakref
from copy import copy
from urllib.parse import urlparse
import aiohttp
import yarl
from fsspec.asyn import AbstractAsyncStreamedFile, AsyncFileSystem, sync, sync_wrapper
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.exceptions import FSTimeoutError
from fsspec.spec import AbstractBufferedFile
from fsspec.utils import (
from ..caching import AllBytes
class HTTPStreamFile(AbstractBufferedFile):

    def __init__(self, fs, url, mode='rb', loop=None, session=None, **kwargs):
        self.asynchronous = kwargs.pop('asynchronous', False)
        self.url = url
        self.loop = loop
        self.session = session
        if mode != 'rb':
            raise ValueError
        self.details = {'name': url, 'size': None}
        super().__init__(fs=fs, path=url, mode=mode, cache_type='none', **kwargs)

        async def cor():
            r = await self.session.get(self.fs.encode_url(url), **kwargs).__aenter__()
            self.fs._raise_not_found_for_status(r, url)
            return r
        self.r = sync(self.loop, cor)

    def seek(self, loc, whence=0):
        if loc == 0 and whence == 1:
            return
        if loc == self.loc and whence == 0:
            return
        raise ValueError('Cannot seek streaming HTTP file')

    async def _read(self, num=-1):
        out = await self.r.content.read(num)
        self.loc += len(out)
        return out
    read = sync_wrapper(_read)

    async def _close(self):
        self.r.close()

    def close(self):
        asyncio.run_coroutine_threadsafe(self._close(), self.loop)
        super().close()

    def __reduce__(self):
        return (reopen, (self.fs, self.url, self.mode, self.blocksize, self.cache.name))