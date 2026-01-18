import functools
import itertools
from typing import Any, NoReturn, Optional, Union, TYPE_CHECKING
from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
def wraps_class_method(method):

    @functools.wraps(method)
    def f(ctype_self: _cuda_types.TypeBase, instance: Data) -> BuiltinFunc:
        return BuiltinFunc.from_class_method(method, ctype_self, instance)
    return f