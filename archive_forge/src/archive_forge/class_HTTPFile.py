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
class HTTPFile(AbstractBufferedFile):
    """
    A file-like object pointing to a remove HTTP(S) resource

    Supports only reading, with read-ahead of a predermined block-size.

    In the case that the server does not supply the filesize, only reading of
    the complete file in one go is supported.

    Parameters
    ----------
    url: str
        Full URL of the remote resource, including the protocol
    session: aiohttp.ClientSession or None
        All calls will be made within this session, to avoid restarting
        connections where the server allows this
    block_size: int or None
        The amount of read-ahead to do, in bytes. Default is 5MB, or the value
        configured for the FileSystem creating this file
    size: None or int
        If given, this is the size of the file in bytes, and we don't attempt
        to call the server to find the value.
    kwargs: all other key-values are passed to requests calls.
    """

    def __init__(self, fs, url, session=None, block_size=None, mode='rb', cache_type='bytes', cache_options=None, size=None, loop=None, asynchronous=False, **kwargs):
        if mode != 'rb':
            raise NotImplementedError('File mode not supported')
        self.asynchronous = asynchronous
        self.url = url
        self.session = session
        self.details = {'name': url, 'size': size, 'type': 'file'}
        super().__init__(fs=fs, path=url, mode=mode, block_size=block_size, cache_type=cache_type, cache_options=cache_options, **kwargs)
        self.loop = loop

    def read(self, length=-1):
        """Read bytes from file

        Parameters
        ----------
        length: int
            Read up to this many bytes. If negative, read all content to end of
            file. If the server has not supplied the filesize, attempting to
            read only part of the data will raise a ValueError.
        """
        if (length < 0 and self.loc == 0) and (not (self.size is not None and self.size <= self.blocksize)):
            self._fetch_all()
        if self.size is None:
            if length < 0:
                self._fetch_all()
        else:
            length = min(self.size - self.loc, length)
        return super().read(length)

    async def async_fetch_all(self):
        """Read whole file in one shot, without caching

        This is only called when position is still at zero,
        and read() is called without a byte-count.
        """
        logger.debug(f'Fetch all for {self}')
        if not isinstance(self.cache, AllBytes):
            r = await self.session.get(self.fs.encode_url(self.url), **self.kwargs)
            async with r:
                r.raise_for_status()
                out = await r.read()
                self.cache = AllBytes(size=len(out), fetcher=None, blocksize=None, data=out)
                self.size = len(out)
    _fetch_all = sync_wrapper(async_fetch_all)

    def _parse_content_range(self, headers):
        """Parse the Content-Range header"""
        s = headers.get('Content-Range', '')
        m = re.match('bytes (\\d+-\\d+|\\*)/(\\d+|\\*)', s)
        if not m:
            return (None, None, None)
        if m[1] == '*':
            start = end = None
        else:
            start, end = [int(x) for x in m[1].split('-')]
        total = None if m[2] == '*' else int(m[2])
        return (start, end, total)

    async def async_fetch_range(self, start, end):
        """Download a block of data

        The expectation is that the server returns only the requested bytes,
        with HTTP code 206. If this is not the case, we first check the headers,
        and then stream the output - if the data size is bigger than we
        requested, an exception is raised.
        """
        logger.debug(f'Fetch range for {self}: {start}-{end}')
        kwargs = self.kwargs.copy()
        headers = kwargs.pop('headers', {}).copy()
        headers['Range'] = f'bytes={start}-{end - 1}'
        logger.debug(f'{self.url} : {headers['Range']}')
        r = await self.session.get(self.fs.encode_url(self.url), headers=headers, **kwargs)
        async with r:
            if r.status == 416:
                return b''
            r.raise_for_status()
            response_is_range = r.status == 206 or self._parse_content_range(r.headers)[0] == start or int(r.headers.get('Content-Length', end + 1)) <= end - start
            if response_is_range:
                out = await r.read()
            elif start > 0:
                raise ValueError("The HTTP server doesn't appear to support range requests. Only reading this file from the beginning is supported. Open with block_size=0 for a streaming file interface.")
            else:
                cl = 0
                out = []
                while True:
                    chunk = await r.content.read(2 ** 20)
                    if chunk:
                        out.append(chunk)
                        cl += len(chunk)
                        if cl > end - start:
                            break
                    else:
                        break
                out = b''.join(out)[:end - start]
            return out
    _fetch_range = sync_wrapper(async_fetch_range)

    def __reduce__(self):
        return (reopen, (self.fs, self.url, self.mode, self.blocksize, self.cache.name if self.cache else 'none', self.size))