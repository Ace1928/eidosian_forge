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
@lru_cache(maxsize=None)
def _dtype_to_storage_type_map():
    return {torch.double: 'DoubleStorage', torch.float: 'FloatStorage', torch.half: 'HalfStorage', torch.long: 'LongStorage', torch.int: 'IntStorage', torch.int16: 'ShortStorage', torch.int8: 'CharStorage', torch.uint8: 'ByteStorage', torch.bool: 'BoolStorage', torch.bfloat16: 'BFloat16Storage', torch.cdouble: 'ComplexDoubleStorage', torch.cfloat: 'ComplexFloatStorage', torch.qint8: 'QInt8Storage', torch.qint32: 'QInt32Storage', torch.quint8: 'QUInt8Storage', torch.quint4x2: 'QUInt4x2Storage', torch.quint2x4: 'QUInt2x4Storage'}