from abc import ABCMeta, abstractmethod, abstractproperty
import contextlib
import errno
import hashlib
import inspect
import itertools
import os
import pickle
import sys
import tempfile
import uuid
import warnings
from numba.misc.appdirs import AppDirs
import numba
from numba.core.errors import NumbaWarning
from numba.core.base import BaseContext
from numba.core.codegen import CodeLibrary
from numba.core.compiler import CompileResult
from numba.core import config, compiler
from numba.core.serialize import dumps
class IndexDataCacheFile(object):
    """
    Implements the logic for the index file and data file used by a cache.
    """

    def __init__(self, cache_path, filename_base, source_stamp):
        self._cache_path = cache_path
        self._index_name = '%s.nbi' % (filename_base,)
        self._index_path = os.path.join(self._cache_path, self._index_name)
        self._data_name_pattern = '%s.{number:d}.nbc' % (filename_base,)
        self._source_stamp = source_stamp
        self._version = numba.__version__

    def flush(self):
        self._save_index({})

    def save(self, key, data):
        """
        Save a new cache entry with *key* and *data*.
        """
        overloads = self._load_index()
        try:
            data_name = overloads[key]
        except KeyError:
            existing = set(overloads.values())
            for i in itertools.count(1):
                data_name = self._data_name(i)
                if data_name not in existing:
                    break
            overloads[key] = data_name
            self._save_index(overloads)
        self._save_data(data_name, data)

    def load(self, key):
        """
        Load a cache entry with *key*.
        """
        overloads = self._load_index()
        data_name = overloads.get(key)
        if data_name is None:
            return
        try:
            return self._load_data(data_name)
        except OSError:
            return

    def _load_index(self):
        """
        Load the cache index and return it as a dictionary (possibly
        empty if cache is empty or obsolete).
        """
        try:
            with open(self._index_path, 'rb') as f:
                version = pickle.load(f)
                data = f.read()
        except FileNotFoundError:
            return {}
        if version != self._version:
            return {}
        stamp, overloads = pickle.loads(data)
        _cache_log('[cache] index loaded from %r', self._index_path)
        if stamp != self._source_stamp:
            return {}
        else:
            return overloads

    def _save_index(self, overloads):
        data = (self._source_stamp, overloads)
        data = self._dump(data)
        with self._open_for_write(self._index_path) as f:
            pickle.dump(self._version, f, protocol=-1)
            f.write(data)
        _cache_log('[cache] index saved to %r', self._index_path)

    def _load_data(self, name):
        path = self._data_path(name)
        with open(path, 'rb') as f:
            data = f.read()
        tup = pickle.loads(data)
        _cache_log('[cache] data loaded from %r', path)
        return tup

    def _save_data(self, name, data):
        data = self._dump(data)
        path = self._data_path(name)
        with self._open_for_write(path) as f:
            f.write(data)
        _cache_log('[cache] data saved to %r', path)

    def _data_name(self, number):
        return self._data_name_pattern.format(number=number)

    def _data_path(self, name):
        return os.path.join(self._cache_path, name)

    def _dump(self, obj):
        return dumps(obj)

    @contextlib.contextmanager
    def _open_for_write(self, filepath):
        """
        Open *filepath* for writing in a race condition-free way (hopefully).
        uuid4 is used to try and avoid name collisions on a shared filesystem.
        """
        uid = uuid.uuid4().hex[:16]
        tmpname = '%s.tmp.%s' % (filepath, uid)
        try:
            with open(tmpname, 'wb') as f:
                yield f
            os.replace(tmpname, filepath)
        except Exception:
            try:
                os.unlink(tmpname)
            except OSError:
                pass
            raise