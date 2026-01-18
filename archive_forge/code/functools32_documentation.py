from _functools import partial, reduce
from collections import MutableMapping, namedtuple
from .reprlib32 import recursive_repr as _recursive_repr
from weakref import proxy as _proxy
import sys as _sys
Clear the cache and cache statistics