from __future__ import annotations
import collections
import enum
import functools
import hashlib
import inspect
import io
import os
import pickle
import sys
import tempfile
import textwrap
import threading
import weakref
from typing import Any, Callable, Dict, Pattern, Type, Union
from streamlit import config, file_util, type_util, util
from streamlit.errors import MarkdownFormattedException, StreamlitAPIException
from streamlit.folder_black_list import FolderBlackList
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit.util import HASHLIB_KWARGS
from, try looking at the hash chain below for an object that you do recognize,
from, try looking at the hash chain below for an object that you do recognize,
class _CodeHasher:
    """A hasher that can hash code objects including dependencies."""

    def __init__(self, hash_funcs: HashFuncsDict | None=None):
        self._hash_funcs: HashFuncsDict
        if hash_funcs:
            self._hash_funcs = {k if isinstance(k, str) else type_util.get_fqn(k): v for k, v in hash_funcs.items()}
        else:
            self._hash_funcs = {}
        self._hashes: dict[Any, bytes] = {}
        self.size = 0

    def __repr__(self) -> str:
        return util.repr_(self)

    def to_bytes(self, obj: Any, context: Context | None=None) -> bytes:
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
            b = b'%s:%s' % (tname, self._to_bytes(obj, context))
            self.size += sys.getsizeof(b)
            if key[1] is not NoResult:
                self._hashes[key] = b
        except (UnhashableTypeError, UserHashError, InternalHashError):
            raise
        except Exception as ex:
            raise InternalHashError(ex, obj)
        finally:
            hash_stacks.current.pop()
        return b

    def update(self, hasher, obj: Any, context: Context | None=None) -> None:
        """Update the provided hasher with the hash of an object."""
        b = self.to_bytes(obj, context)
        hasher.update(b)

    def _file_should_be_hashed(self, filename: str) -> bool:
        global _FOLDER_BLACK_LIST
        if not _FOLDER_BLACK_LIST:
            _FOLDER_BLACK_LIST = FolderBlackList(config.get_option('server.folderWatchBlacklist'))
        filepath = os.path.abspath(filename)
        file_is_blacklisted = _FOLDER_BLACK_LIST.is_blacklisted(filepath)
        if file_is_blacklisted:
            return False
        return file_util.file_is_in_folder_glob(filepath, self._get_main_script_directory()) or file_util.file_in_pythonpath(filepath)

    def _to_bytes(self, obj: Any, context: Context | None) -> bytes:
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
                raise UserHashError(ex, obj, hash_func=hash_func)
            return self.to_bytes(output)
        elif isinstance(obj, str):
            return obj.encode()
        elif isinstance(obj, float):
            return self.to_bytes(hash(obj))
        elif isinstance(obj, int):
            return _int_to_bytes(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self.update(h, item, context)
            return h.digest()
        elif isinstance(obj, dict):
            for item in obj.items():
                self.update(h, item, context)
            return h.digest()
        elif obj is None:
            return b'0'
        elif obj is True:
            return b'1'
        elif obj is False:
            return b'0'
        elif type_util.is_type(obj, 'pandas.core.frame.DataFrame') or type_util.is_type(obj, 'pandas.core.series.Series'):
            import pandas as pd
            if len(obj) >= _PANDAS_ROWS_LARGE:
                obj = obj.sample(n=_PANDAS_SAMPLE_SIZE, random_state=0)
            try:
                return b'%s' % pd.util.hash_pandas_object(obj).sum()
            except TypeError:
                return b'%s' % pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
        elif type_util.is_type(obj, 'numpy.ndarray'):
            self.update(h, obj.shape)
            if obj.size >= _NP_SIZE_LARGE:
                import numpy as np
                state = np.random.RandomState(0)
                obj = state.choice(obj.flat, size=_NP_SAMPLE_SIZE)
            self.update(h, obj.tobytes())
            return h.digest()
        elif inspect.isbuiltin(obj):
            return bytes(obj.__name__.encode())
        elif any((type_util.is_type(obj, typename) for typename in _FFI_TYPE_NAMES)):
            return self.to_bytes(None)
        elif type_util.is_type(obj, 'builtins.mappingproxy') or type_util.is_type(obj, 'builtins.dict_items'):
            return self.to_bytes(dict(obj))
        elif type_util.is_type(obj, 'builtins.getset_descriptor'):
            return bytes(obj.__qualname__.encode())
        elif isinstance(obj, UploadedFile):
            h = hashlib.new('md5', **HASHLIB_KWARGS)
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
        elif any((type_util.get_fqn(x) == 'sqlalchemy.pool.base.Pool' for x in type(obj).__bases__)):
            cargs = obj._creator.__closure__
            cargs = [cargs[0].cell_contents, cargs[1].cell_contents] if cargs else None
            if cargs:
                cargs[1] = dict(collections.OrderedDict(sorted(cargs[1].items(), key=lambda t: t[0])))
            reduce_data = obj.__reduce__()
            for attr in ['_overflow_lock', '_pool', '_conn', '_fairy', '_threadconns', 'logger']:
                reduce_data[2].pop(attr, None)
            return self.to_bytes([reduce_data, cargs])
        elif type_util.is_type(obj, 'sqlalchemy.engine.base.Engine'):
            reduce_data = obj.__reduce__()
            reduce_data[2].pop('url', None)
            reduce_data[2].pop('logger', None)
            return self.to_bytes(reduce_data)
        elif type_util.is_type(obj, 'numpy.ufunc'):
            return bytes(obj.__name__.encode())
        elif type_util.is_type(obj, 'socket.socket'):
            return self.to_bytes(id(obj))
        elif any((type_util.get_fqn(x) == 'torch.nn.modules.module.Module' for x in type(obj).__bases__)):
            return self.to_bytes(id(obj))
        elif type_util.is_type(obj, 'tensorflow.python.client.session.Session'):
            return self.to_bytes(id(obj))
        elif type_util.is_type(obj, 'torch.Tensor') or type_util.is_type(obj, 'torch._C._TensorBase'):
            return self.to_bytes([obj.detach().numpy(), obj.grad])
        elif any((type_util.is_type(obj, typename) for typename in _KERAS_TYPE_NAMES)):
            return self.to_bytes(id(obj))
        elif type_util.is_type(obj, 'tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject'):
            return self.to_bytes(id(obj))
        elif inspect.isroutine(obj):
            wrapped = getattr(obj, '__wrapped__', None)
            if wrapped is not None:
                return self.to_bytes(wrapped)
            if obj.__module__.startswith('streamlit'):
                return self.to_bytes('{}.{}'.format(obj.__module__, obj.__name__))
            code = getattr(obj, '__code__', None)
            assert code is not None
            if self._file_should_be_hashed(code.co_filename):
                context = _get_context(obj)
                defaults = getattr(obj, '__defaults__', None)
                if defaults is not None:
                    self.update(h, defaults, context)
                h.update(self._code_to_bytes(code, context, func=obj))
            else:
                self.update(h, obj.__module__)
                self.update(h, obj.__name__)
            return h.digest()
        elif inspect.iscode(obj):
            if context is None:
                raise RuntimeError('context must be defined when hashing code')
            return self._code_to_bytes(obj, context)
        elif inspect.ismodule(obj):
            return self.to_bytes(obj.__name__)
        elif inspect.isclass(obj):
            return self.to_bytes(obj.__name__)
        elif isinstance(obj, functools.partial):
            h = hashlib.new('md5', **HASHLIB_KWARGS)
            self.update(h, obj.args)
            self.update(h, obj.func)
            self.update(h, obj.keywords)
            return h.digest()
        else:
            try:
                reduce_data = obj.__reduce__()
            except Exception as ex:
                raise UnhashableTypeError(ex, obj)
            for item in reduce_data:
                self.update(h, item, context)
            return h.digest()

    def _code_to_bytes(self, code, context: Context, func=None) -> bytes:
        h = hashlib.new('md5', **HASHLIB_KWARGS)
        self.update(h, code.co_code)
        consts = [n for n in code.co_consts if not isinstance(n, str) or not n.endswith('.<lambda>')]
        self.update(h, consts, context)
        context.cells.push(code, func=func)
        for ref in get_referenced_objects(code, context):
            self.update(h, ref, context)
        context.cells.pop()
        return h.digest()

    @staticmethod
    def _get_main_script_directory() -> str:
        """Get the absolute path to directory of the main script."""
        import pathlib
        import __main__
        abs_main_path = pathlib.Path(__main__.__file__).resolve()
        return str(abs_main_path.parent)