import contextlib
import dataclasses
import functools
import itertools
import logging
import operator
import re
from collections import namedtuple
from itertools import chain
from typing import (
import sympy
from sympy.printing.printer import Printer
import torch
import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges
from .. import config, metrics
from ..utils import (
from ..virtualized import ops, OpsValue, V
class CSEVariable:
    """A CSEVariable is just a name for an expression but it is useful to be able to annotate them on a backend dependent basis.
    To do so, the backends can simply overload `Kernel.create_cse_var`
    The "CSEVariable.update_on_args" method gives you a hook for annotations
    See example of TritonCSEVariable in triton.py
    """

    def __init__(self, name, bounds: ValueRanges):
        assert isinstance(bounds, ValueRanges)
        self.name = name
        self.bounds = bounds

    def __str__(self):
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return type(other) == type(self) and other.name == self.name

    def update_on_args(self, name, args, kwargs):
        pass