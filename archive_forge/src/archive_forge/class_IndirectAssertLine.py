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
class IndirectAssertLine(DeferredLineBase):

    def __init__(self, line, assert_fn, var, mask, size_map):
        self.var = var
        self.mask = mask
        self.line = line
        self.assert_fn = assert_fn
        self.size_map = size_map

    def __call__(self):
        size, size_str = self.size_map[self.var, self.mask]
        assert_min = (self.var.bounds.lower >= 0) != sympy.true
        assert_max = (self.var.bounds.upper < size) != sympy.true
        if not (assert_min or assert_max):
            return None
        elif assert_min and assert_max:
            cond = f'(0 <= {self.var}) & ({self.var} < {size_str})'
            cond_print = f'0 <= {self.var} < {size_str}'
        elif assert_min:
            cond = f'0 <= {self.var}'
            cond_print = cond
        else:
            assert assert_max
            cond = f'{self.var} < {size_str}'
            cond_print = cond
        if self.mask:
            cond = f'({cond}) | ~{self.mask}'
        return self.line.format(assert_fn=self.assert_fn, cond=cond, cond_print=cond_print)

    def _new_line(self, line):
        return IndirectAssertLine(line, self.assert_fn, self.var, self.mask, self.size_map)