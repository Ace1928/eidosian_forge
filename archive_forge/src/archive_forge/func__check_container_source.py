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
def _check_container_source(container_type, source_file, original_source):
    try:
        current_source = ''.join(get_source_lines_and_file(container_type)[0])
    except Exception:
        warnings.warn("Couldn't retrieve source code for container of type " + container_type.__name__ + ". It won't be checked for correctness upon loading.")
        return
    if original_source != current_source:
        if container_type.dump_patches:
            file_name = container_type.__name__ + '.patch'
            diff = difflib.unified_diff(current_source.split('\n'), original_source.split('\n'), source_file, source_file, lineterm='')
            lines = '\n'.join(diff)
            try:
                with open(file_name, 'a+') as f:
                    file_size = f.seek(0, 2)
                    f.seek(0)
                    if file_size == 0:
                        f.write(lines)
                    elif file_size != len(lines) or f.read() != lines:
                        raise OSError
                msg = 'Saved a reverse patch to ' + file_name + '. Run `patch -p0 < ' + file_name + '` to revert your changes.'
            except OSError:
                msg = "Tried to save a patch, but couldn't create a writable file " + file_name + ". Make sure it doesn't exist and your working directory is writable."
        else:
            msg = "you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes."
        msg = f"source code of class '{torch.typename(container_type)}' has changed. {msg}"
        warnings.warn(msg, SourceChangeWarning)