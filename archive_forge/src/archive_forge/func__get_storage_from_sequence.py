import io
import torch
from ._utils import _type, _cuda, _hpu
from torch.types import Storage
from typing import cast, Any, Dict as _Dict, Optional as _Optional, TypeVar, Type, Union
import copy
import collections
from functools import lru_cache
import warnings
import threading
import functools
def _get_storage_from_sequence(sequence, dtype, device):
    if dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
        interpret_dtypes = {torch.quint8: torch.uint8, torch.quint4x2: torch.uint8, torch.quint2x4: torch.uint8, torch.qint32: torch.int32, torch.qint8: torch.int8}
        tmp_tensor = torch.tensor(sequence, dtype=interpret_dtypes[dtype], device=device)
    else:
        tmp_tensor = torch.tensor(sequence, dtype=dtype, device=device)
    return tmp_tensor._typed_storage()._untyped_storage