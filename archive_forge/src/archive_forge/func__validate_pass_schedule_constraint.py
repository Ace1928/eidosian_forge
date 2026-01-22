import inspect
import logging
from queue import Queue
from functools import wraps
from typing import Callable, Dict, List
import torch.nn as nn
from torch.fx.graph_module import GraphModule
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult
def _validate_pass_schedule_constraint(constraint: Callable[[Callable, Callable], bool], passes: List[Callable]) -> None:
    for i, a in enumerate(passes):
        for j, b in enumerate(passes[i + 1:]):
            if constraint(a, b):
                continue
            raise RuntimeError(f'pass schedule constraint violated. Expected {a} before {b} but found {a} at index {i} and {b} at index{j} in pass list.')