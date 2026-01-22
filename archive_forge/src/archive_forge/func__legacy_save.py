import difflib
import os
import io
import shutil
import struct
import sys
import torch
import tarfile
import tempfile
import warnings
from contextlib import closing, contextmanager
from enum import Enum
from ._utils import _import_dotted_name
from torch._sources import get_source_lines_and_file
from torch.types import Storage
from torch.storage import _get_dtype_from_pickle_storage_type
from typing import Any, BinaryIO, Callable, cast, Dict, Optional, Type, Tuple, Union, IO
from typing_extensions import TypeAlias  # Python 3.10+
import copyreg
import pickle
import pathlib
import torch._weights_only_unpickler as _weights_only_unpickler
def _legacy_save(obj, f, pickle_module, pickle_protocol) -> None:
    import torch.nn as nn
    serialized_container_types = {}
    serialized_storages = {}
    storage_dtypes: Dict[int, torch.dtype] = {}

    def persistent_id(obj: Any) -> Optional[Tuple]:
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            if obj in serialized_container_types:
                return None
            serialized_container_types[obj] = True
            source_file = source = None
            try:
                source_lines, _, source_file = get_source_lines_and_file(obj)
                source = ''.join(source_lines)
            except Exception:
                warnings.warn("Couldn't retrieve source code for container of type " + obj.__name__ + ". It won't be checked for correctness upon loading.")
            return ('module', obj, source_file, source)
        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            storage: torch.UntypedStorage
            if isinstance(obj, torch.storage.TypedStorage):
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                dtype = obj.dtype
                storage_numel = obj._size()
            elif isinstance(obj, torch.UntypedStorage):
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                dtype = torch.uint8
                storage_numel = storage.nbytes()
            else:
                raise TypeError(f'type not recognized: {type(obj)}')
            if storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError('Cannot save multiple tensors or storages that view the same data as different types')
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype
            view_metadata: Optional[Tuple[str, int, int]]
            offset = 0
            storage_key = str(storage._cdata)
            location = location_tag(storage)
            if storage_key not in serialized_storages:
                serialized_storages[storage_key] = (storage, dtype)
            is_view = storage._cdata != storage._cdata
            if is_view:
                view_metadata = (str(storage._cdata), offset, storage.nbytes())
            else:
                view_metadata = None
            res = ('storage', storage_type, storage_key, location, storage_numel, view_metadata)
            return res
        return None
    sys_info = dict(protocol_version=PROTOCOL_VERSION, little_endian=sys.byteorder == 'little', type_sizes=dict(short=SHORT_SIZE, int=INT_SIZE, long=LONG_SIZE))
    pickle_module.dump(MAGIC_NUMBER, f, protocol=pickle_protocol)
    pickle_module.dump(PROTOCOL_VERSION, f, protocol=pickle_protocol)
    pickle_module.dump(sys_info, f, protocol=pickle_protocol)
    pickler = pickle_module.Pickler(f, protocol=pickle_protocol)
    pickler.persistent_id = persistent_id
    pickler.dump(obj)
    serialized_storage_keys = sorted(serialized_storages.keys())
    pickle_module.dump(serialized_storage_keys, f, protocol=pickle_protocol)
    f.flush()
    for key in serialized_storage_keys:
        storage, dtype = serialized_storages[key]
        storage._write_file(f, _should_read_directly(f), True, torch._utils._element_size(dtype))