import inspect
import operator
import types as pytypes
import typing as pt
from collections import OrderedDict
from collections.abc import Sequence
from llvmlite import ir as llvmir
from numba import njit
from numba.core import cgutils, errors, imputils, types, utils
from numba.core.datamodel import default_manager, models
from numba.core.registry import cpu_target
from numba.core.typing import templates
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.serialize import disable_pickling
from numba.experimental.jitclass import _box
def _drop_ignored_attrs(dct):
    drop = set(['__weakref__', '__module__', '__dict__'])
    if '__annotations__' in dct:
        drop.add('__annotations__')
    for k, v in dct.items():
        if isinstance(v, (pytypes.BuiltinFunctionType, pytypes.BuiltinMethodType)):
            drop.add(k)
        elif getattr(v, '__objclass__', None) is object:
            drop.add(k)
    if '__hash__' in dct and dct['__hash__'] is None:
        drop.add('__hash__')
    for k in drop:
        del dct[k]