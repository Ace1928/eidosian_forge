import builtins
import collections
import copy
import dataclasses
import functools
import inspect
import itertools
import math
import operator
import sys
import types
import warnings
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Set, Union
import torch
import torch._functorch.deprecated as deprecated_func
from torch.fx._symbolic_trace import is_fx_tracing
from . import config
from .external_utils import is_compiling
from .utils import hashable, is_safe_constant, NP_SUPPORTED_MODULES
class FunctionIdSet:
    """
    Track a set of `id()`s of objects which are either allowed or not
    allowed to go into the generated FX graph.  Use to test for torch.*,
    numpy.*, builtins.*, etc.

    Support user modification to permit customization of what can be
    added to the graph and what will cause a graph break.
    """
    function_ids: Optional[Set[int]] = None
    function_names: Optional[Dict[int, str]] = None

    def __init__(self, lazy_initializer: Callable[[], Union[Dict[int, str], Set[int]]]):
        self.lazy_initializer = lazy_initializer

    def __call__(self):
        if self.function_ids is None:
            value = self.lazy_initializer()
            if isinstance(value, dict):
                self.function_ids = set(value.keys())
                self.function_names = value
            else:
                assert isinstance(value, set)
                self.function_ids = value
        return self.function_ids

    def get_name(self, idx: int, default: str):
        self()
        assert self.function_names is not None
        return self.function_names.get(idx, default)

    def add(self, idx: int):
        function_ids = self()
        function_ids.add(idx)

    def remove(self, idx: int):
        function_ids = self()
        if idx in function_ids:
            function_ids.remove(idx)

    def __contains__(self, idx: int):
        return idx in self()