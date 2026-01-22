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
@staticmethod
def _spec_of(arg):
    if hasattr(arg, 'data_ptr'):
        return arg.data_ptr() % JITFunction.divisibility == 0
    elif isinstance(arg, int):
        return (arg % 16 == 0, arg == 1)
    return (arg is None,)