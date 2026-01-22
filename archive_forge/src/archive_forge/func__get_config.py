from __future__ import annotations, division
import ast
import functools
import hashlib
import inspect
import os
import textwrap
from collections import defaultdict, namedtuple
from functools import cached_property
from typing import Callable, Generic, Iterable, List, Optional, TypeVar, Union, cast, overload
from .._C.libtriton.triton import TMAInfos
from ..common.backend import get_backend, get_cuda_version_key
from .interpreter import InterpretedFunction
def _get_config(self, *args):

    def is_divisible_by_16(x):
        if hasattr(x, 'data_ptr'):
            return x.data_ptr() % JITFunction.divisibility == 0
        elif isinstance(x, int):
            return x % JITFunction.divisibility == 0
        if x is None:
            return True
        return False

    def is_divisible_by_8(x):
        if isinstance(x, int):
            return x % JITFunction.divisibility_8 == 0
        if x is None:
            return True
        return False
    divisible_by_16 = {param.num for param, arg in zip(self.params, args) if is_divisible_by_16(arg) and (not param.do_not_specialize)}
    divisible_by_8 = {param.num for param, arg in zip(self.params, args) if is_divisible_by_8(arg) and (not param.do_not_specialize)}
    equal_to_1 = {param.num for param, arg in zip(self.params, args) if isinstance(arg, int) and (not isinstance(arg, bool)) and (arg == 1) and (not param.do_not_specialize)}
    none_args = {param.num for param, arg in zip(self.params, args) if arg is None and (not param.do_not_specialize)}
    ids_of_folded_args = equal_to_1 | none_args
    return namedtuple('instance_descriptor', ['divisible_by_16', 'equal_to_1', 'ids_of_folded_args', 'divisible_by_8'])(tuple(divisible_by_16), tuple(equal_to_1), tuple(ids_of_folded_args), tuple(divisible_by_8))