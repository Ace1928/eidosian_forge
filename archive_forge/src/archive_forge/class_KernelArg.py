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
class KernelArg:
    """Represents an argument to a @jit'ed function.

    An argument is a parameter plus a value.
    """

    def __init__(self, value, param):
        self.value = value
        self.param = param

    @property
    def name(self):
        return self.param.name

    def signature_key(self):
        annotation = self.param.annotation
        if 'Tensor' in annotation:
            return self.value.dtype
        elif annotation == 'bool':
            return 'i1'
        elif annotation == 'float':
            return 'fp32'
        else:
            return JITFunction._key_of(self.value)

    def specialization_key(self):
        assert not self.param.do_not_specialize
        try:
            return (self.value.data_ptr() % JITFunction.divisibility == 0,)
        except AttributeError:
            pass
        if isinstance(self.value, int):
            return (self.value % JITFunction.divisibility == 0, self.value % JITFunction.divisibility_8 == 0, self.value == 1)
        return (False,)