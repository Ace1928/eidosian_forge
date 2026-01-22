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
def _privateuse1_deserialize(obj, location):
    backend_name = torch._C._get_privateuse1_backend_name()
    if location.startswith(backend_name):
        if not hasattr(obj, backend_name):
            raise RuntimeError(f'Attempting to load the storages to the {backend_name.upper()} device but torch.storage._StorageBase.{backend_name}() or torch.storage.TypedStorage.{backend_name}() is not generated. Please use torch.utils.generate_methods_for_privateuse1_backend to generate storage.{backend_name}() method first.')
        device_index = _validate_privateuse1_device(location, backend_name)
        return getattr(obj, backend_name)(device_index)