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
def _set_handler(self, value):
    prior = self._get_handler()
    setattr(threadlocal, self._key, value)

    @contextmanager
    def ctx():
        try:
            yield
        finally:
            self._set_handler(prior)
    return ctx()