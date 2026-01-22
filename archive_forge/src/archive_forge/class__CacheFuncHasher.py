from __future__ import annotations
import collections
import dataclasses
import datetime
import functools
import hashlib
import inspect
import io
import os
import pickle
import sys
import tempfile
import threading
import uuid
import weakref
from enum import Enum
from typing import Any, Callable, Dict, Final, Pattern, Type, Union
from typing_extensions import TypeAlias
from streamlit import type_util, util
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.caching.cache_errors import UnhashableTypeError
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit.util import HASHLIB_KWARGS
class _CacheFuncHasher:
    """A hasher that can hash objects with cycles."""

    def __init__(self, cache_type: CacheType, hash_funcs: HashFuncsDict | None=None):
        self._hash_funcs: HashFuncsDict
        if hash_funcs:
            self._hash_funcs = {k if isinstance(k, str) else type_util.get_fqn(k): v for k, v in hash_funcs.items()}
        else:
            self._hash_funcs = {}
        self._hashes: dict[Any, bytes] = {}
        self.size = 0
        self.cache_type = cache_type

    def __repr__(self) -> str:
        return util.repr_(self)

    def to_bytes(self, obj: Any) -> bytes:
        """Add memoization to _to_bytes and protect against cycles in data structures."""
        tname = type(obj).__qualname__.encode()
        key = (tname, _key(obj))
        if key[1] is not NoResult:
            if key in self._hashes:
                return self._hashes[key]
        if obj in hash_stacks.current:
            return _CYCLE_PLACEHOLDER
        hash_stacks.current.push(obj)
        try:
            b = b'%s:%s' % (tname, self._to_bytes(obj))
            self.size += sys.getsizeof(b)
            if key[1] is not NoResult:
                self._hashes[key] = b
        finally:
            hash_stacks.current.pop()
        return b

    def update(self, hasher, obj: Any) -> None:
        """Update the provided hasher with the hash of an object."""
        b = self.to_bytes(obj)
        hasher.update(b)

    def _to_bytes(self, obj: Any) -> bytes:
        """Hash objects to bytes, including code with dependencies.

        Python's built in `hash` does not produce consistent results across
        runs.
        """
        h = hashlib.new('md5', **HASHLIB_KWARGS)
        if type_util.is_type(obj, 'unittest.mock.Mock') or type_util.is_type(obj, 'unittest.mock.MagicMock'):
            return self.to_bytes(id(obj))
        elif isinstance(obj, bytes) or isinstance(obj, bytearray):
            return obj
        elif type_util.get_fqn_type(obj) in self._hash_funcs:
            hash_func = self._hash_funcs[type_util.get_fqn_type(obj)]
            try:
                output = hash_func(obj)
            except Exception as ex:
                raise UserHashError(ex, obj, hash_func=hash_func, cache_type=self.cache_type) from ex
            return self.to_bytes(output)
        elif isinstance(obj, str):
            return obj.encode()
        elif isinstance(obj, float):
            return _float_to_bytes(obj)
        elif isinstance(obj, int):
            return _int_to_bytes(obj)
        elif isinstance(obj, uuid.UUID):
            return obj.bytes
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat().encode()
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self.update(h, item)
            return h.digest()
        elif isinstance(obj, dict):
            for item in obj.items():
                self.update(h, item)
            return h.digest()
        elif obj is None:
            return b'0'
        elif obj is True:
            return b'1'
        elif obj is False:
            return b'0'
        elif dataclasses.is_dataclass(obj):
            return self.to_bytes(dataclasses.asdict(obj))
        elif isinstance(obj, Enum):
            return str(obj).encode()
        elif type_util.is_type(obj, 'pandas.core.series.Series'):
            import pandas as pd
            self.update(h, obj.size)
            self.update(h, obj.dtype.name)
            if len(obj) >= _PANDAS_ROWS_LARGE:
                obj = obj.sample(n=_PANDAS_SAMPLE_SIZE, random_state=0)
            try:
                self.update(h, pd.util.hash_pandas_object(obj).values.tobytes())
                return h.digest()
            except TypeError:
                return b'%s' % pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
        elif type_util.is_type(obj, 'pandas.core.frame.DataFrame'):
            import pandas as pd
            self.update(h, obj.shape)
            if len(obj) >= _PANDAS_ROWS_LARGE:
                obj = obj.sample(n=_PANDAS_SAMPLE_SIZE, random_state=0)
            try:
                column_hash_bytes = self.to_bytes(pd.util.hash_pandas_object(obj.dtypes))
                self.update(h, column_hash_bytes)
                values_hash_bytes = self.to_bytes(pd.util.hash_pandas_object(obj))
                self.update(h, values_hash_bytes)
                return h.digest()
            except TypeError:
                return b'%s' % pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
        elif type_util.is_type(obj, 'numpy.ndarray'):
            self.update(h, obj.shape)
            self.update(h, str(obj.dtype))
            if obj.size >= _NP_SIZE_LARGE:
                import numpy as np
                state = np.random.RandomState(0)
                obj = state.choice(obj.flat, size=_NP_SAMPLE_SIZE)
            self.update(h, obj.tobytes())
            return h.digest()
        elif type_util.is_type(obj, 'PIL.Image.Image'):
            import numpy as np
            np_array = np.frombuffer(obj.tobytes(), dtype='uint8')
            return self.to_bytes(np_array)
        elif inspect.isbuiltin(obj):
            return bytes(obj.__name__.encode())
        elif type_util.is_type(obj, 'builtins.mappingproxy') or type_util.is_type(obj, 'builtins.dict_items'):
            return self.to_bytes(dict(obj))
        elif type_util.is_type(obj, 'builtins.getset_descriptor'):
            return bytes(obj.__qualname__.encode())
        elif isinstance(obj, UploadedFile):
            self.update(h, obj.name)
            self.update(h, obj.tell())
            self.update(h, obj.getvalue())
            return h.digest()
        elif hasattr(obj, 'name') and (isinstance(obj, io.IOBase) or isinstance(obj, tempfile._TemporaryFileWrapper)):
            obj_name = getattr(obj, 'name', 'wonthappen')
            self.update(h, obj_name)
            self.update(h, os.path.getmtime(obj_name))
            self.update(h, obj.tell())
            return h.digest()
        elif isinstance(obj, Pattern):
            return self.to_bytes([obj.pattern, obj.flags])
        elif isinstance(obj, io.StringIO) or isinstance(obj, io.BytesIO):
            self.update(h, obj.tell())
            self.update(h, obj.getvalue())
            return h.digest()
        elif type_util.is_type(obj, 'numpy.ufunc'):
            return bytes(obj.__name__.encode())
        elif inspect.ismodule(obj):
            return self.to_bytes(obj.__name__)
        elif inspect.isclass(obj):
            return self.to_bytes(obj.__name__)
        elif isinstance(obj, functools.partial):
            self.update(h, obj.args)
            self.update(h, obj.func)
            self.update(h, obj.keywords)
            return h.digest()
        else:
            try:
                reduce_data = obj.__reduce__()
            except Exception as ex:
                raise UnhashableTypeError() from ex
            for item in reduce_data:
                self.update(h, item)
            return h.digest()