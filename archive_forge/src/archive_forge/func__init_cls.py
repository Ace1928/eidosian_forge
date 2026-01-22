from __future__ import annotations
import itertools
from contextlib import contextmanager
from itertools import chain
from threading import local
from typing import Any, Callable, TYPE_CHECKING, Union
from unittest.mock import patch
import sympy
from torch._inductor.utils import IndentedBuffer
from torch.fx.graph import inplace_methods, magic_methods
from .utils import reduction_num_outputs, sympy_str, sympy_symbol
@classmethod
def _init_cls(cls):

    def make_handler(format_string):

        @staticmethod
        def inner(*args):
            return format_string.format(*args)
        return inner
    for name, format_string in chain(magic_methods.items(), inplace_methods.items()):
        setattr(cls, name, make_handler(format_string))