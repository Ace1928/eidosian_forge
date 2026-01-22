from __future__ import annotations
import os
import io
import abc
import zlib
import errno
import time
import struct
import sqlite3
import threading
import pickletools
import asyncio
import inspect
import dill as pkl
import functools as ft
import contextlib as cl
import warnings
from fileio.lib.types import File, FileLike
from typing import Any, Optional, Type, Dict, Union, Tuple, TYPE_CHECKING
from lazyops.utils.pooler import ThreadPooler
from lazyops.libs.sqlcache.constants import (
from lazyops.libs.sqlcache.config import SqlCacheConfig
from lazyops.libs.sqlcache.exceptions import (
from lazyops.libs.sqlcache.utils import (
class BaseMedium(abc.ABC):
    pickle_protocol: int = 3
    '\n    Base Medium Class for SqlCache\n    '

    def serialize(self, value: ValueT, protocol: Optional[Any]=None, optimize: Optional[bool]=False, **kwargs) -> bytes:
        """
        Serialize value to bytes
        """
        protocol = protocol or self.pickle_protocol
        data = pkl.dumps(value, protocol=self.pickle_protocol)
        if optimize:
            data = pickletools.optimize(data)
        return data

    async def aserialize(self, value: ValueT, protocol: Optional[Any]=None, optimize: Optional[bool]=False, **kwargs) -> bytes:
        """
        Serialize value to bytes
        """
        return await ThreadPooler.run_async(self.serialize, value, protocol, optimize, **kwargs)

    def deserialize(self, data: bytes, **kwargs) -> ValueT:
        """
        Deserialize bytes to value
        """
        return pkl.loads(data)

    async def adeserialize(self, data: bytes, **kwargs) -> ValueT:
        """
        Deserialize bytes to value
        """
        return await ThreadPooler.run_async(self.deserialize, data, **kwargs)

    def hash(self, key: KeyT):
        """Compute portable hash for `key`.
        :param key: key to hash
        :return: hash value
        """
        mask = 4294967295
        disk_key, _ = self.put(key)
        type_disk_key = type(disk_key)
        if type_disk_key is sqlite3.Binary:
            return zlib.adler32(disk_key) & mask
        elif type_disk_key is str:
            return zlib.adler32(disk_key.encode('utf-8')) & mask
        elif type_disk_key is int:
            return disk_key % mask
        else:
            assert type_disk_key is float
            return zlib.adler32(struct.pack('!d', disk_key)) & mask

    async def ahash(self, key: KeyT):
        """Compute portable hash for `key`.
        :param key: key to hash
        :return: hash value
        """
        return await ThreadPooler.run_async(self.hash, key)

    def put(self, key: KeyT):
        """Convert `key` to fields key and raw for Store table.
        :param key: key to convert
        :return: (database key, raw boolean) pair
        """
        type_key = type(key)
        if type_key is bytes:
            return (sqlite3.Binary(key), True)
        elif type_key is str or (type_key is int and -9223372036854775808 <= key <= 9223372036854775807) or type_key is float:
            return (key, True)
        else:
            result = self.serialize(key, optimize=True)
            return (sqlite3.Binary(result), False)

    async def aput(self, key: KeyT):
        """Convert `key` to fields key and raw for Store table.
        :param key: key to convert
        :return: (database key, raw boolean) pair
        """
        return await ThreadPooler.run_async(self.put, key)

    def get(self, key: KeyT, raw: bool):
        """Convert fields `key` and `raw` from Store table to key.
        :param key: database key to convert
        :param bool raw: flag indicating raw database storage
        :return: corresponding Python key
        """
        if raw:
            return bytes(key) if type(key) is sqlite3.Binary else key
        else:
            return self.deserialize(io.BytesIO(key))

    async def aget(self, key: KeyT, raw: bool):
        """Convert fields `key` and `raw` from Store table to key.
        :param key: database key to convert
        :param bool raw: flag indicating raw database storage
        :return: corresponding Python key
        """
        if raw:
            return bytes(key) if type(key) is sqlite3.Binary else key
        else:
            return await self.adeserialize(io.BytesIO(key))

    def store(self, value: ValueT, read: bool, key: KeyT=UNKNOWN):
        """Convert `value` to fields size, mode, filename, and value for Store
        table.
        :param value: value to convert
        :param bool read: True when value is file-like object
        :param key: key for item (default UNKNOWN)
        :return: (size, mode, filename, value) tuple for Store table
        """
        raise NotImplementedError

    async def astore(self, value: ValueT, read: bool, key: KeyT=UNKNOWN):
        """Convert `value` to fields size, mode, filename, and value for Store
        table.
        :param value: value to convert
        :param bool read: True when value is file-like object
        :param key: key for item (default UNKNOWN)
        :return: (size, mode, filename, value) tuple for Store table
        """
        raise NotImplementedError

    def _write(self, *args, **kwargs):
        """Write `value` to file-like object `file`.
        :param file: file-like object
        :param value: value to write
        """
        raise NotImplementedError

    async def _awrite(self, *args, **kwargs):
        """Write `value` to file-like object `file`.
        :param file: file-like object
        :param value: value to write
        """
        raise NotImplementedError

    def fetch(self, *args, **kwargs):
        """Fetch value from file-like object `file`.
        :param file: file-like object
        :return: value
        """
        raise NotImplementedError

    async def afetch(self, *args, **kwargs):
        """Fetch value from file-like object `file`.
        :param file: file-like object
        :return: value
        """
        raise NotImplementedError

    def filename(self, key: KeyT=UNKNOWN, value: ValueT=UNKNOWN):
        """Return filename and full-path tuple for file storage.
        Filename will be a randomly generated 28 character hexadecimal string
        with ".val" suffixed. Two levels of sub-directories will be used to
        reduce the size of directories. On older filesystems, lookups in
        directories with many files may be slow.
        The default implementation ignores the `key` and `value` parameters.
        In some scenarios, for example :meth:`Store.push
        <diskcache.Store.push>`, the `key` or `value` may not be known when the
        item is stored in the cache.
        :param key: key for item (default UNKNOWN)
        :param value: value for item (default UNKNOWN)
        """
        raise NotImplementedError

    def remove(self, *args, **kwargs):
        """Remove a file given by `filename`.
        This method is cross-thread and cross-process safe. If an "error no
        entry" occurs, it is suppressed.
        :param str filename: relative path to file
        """
        raise NotImplementedError

    async def aremove(self, *args, **kwargs):
        """Remove a file given by `filename`.
        This method is cross-thread and cross-process safe. If an "error no
        entry" occurs, it is suppressed.
        :param str filename: relative path to file
        """
        raise NotImplementedError