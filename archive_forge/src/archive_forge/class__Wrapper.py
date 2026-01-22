import functools
import itertools
from typing import Any, NoReturn, Optional, Union, TYPE_CHECKING
from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
class _Wrapper(BuiltinFunc):

    def call(self, env, *args, **kwargs):
        return method(ctype_self, env, instance, *args)