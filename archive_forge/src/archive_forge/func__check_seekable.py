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
def _check_seekable(f) -> bool:

    def raise_err_msg(patterns, e):
        for p in patterns:
            if p in str(e):
                msg = str(e) + '. You can only torch.load from a file that is seekable.' + ' Please pre-load the data into a buffer like io.BytesIO and' + ' try to load from it instead.'
                raise type(e)(msg)
        raise e
    try:
        f.seek(f.tell())
        return True
    except (io.UnsupportedOperation, AttributeError) as e:
        raise_err_msg(['seek', 'tell'], e)
    return False