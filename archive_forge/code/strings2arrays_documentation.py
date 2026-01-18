from ctypes import c_uint64
from typing import Callable, List, Sequence, Tuple
from murmurhash import hash_unicode
from ..config import registry
from ..model import Model
from ..types import Ints2d
Transform a sequence of string sequences to a list of arrays.