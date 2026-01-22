import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
def _set_parameters_using_python_tracer(self) -> None:
    for event in self._op_tree.dfs():
        for p in extract_parameters(event):
            if p is not None:
                self._categories.set_by_id(p, Category.PARAMETER)