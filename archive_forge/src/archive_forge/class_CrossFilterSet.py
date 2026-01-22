import sys
import weakref
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
from .core import util
from .core.ndmapping import UniformNdMapping
class CrossFilterSet(Derived):
    selection_expr = param.Parameter(default=None, constant=True)

    def __init__(self, selection_streams=(), mode='intersection', index_cols=None, **params):
        self._mode = mode
        self._index_cols = index_cols
        input_streams = list(selection_streams)
        exclusive = mode == 'overwrite'
        super().__init__(input_streams, exclusive=exclusive, **params)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, v):
        if v != self._mode:
            self._mode = v
            self.reset()
            self.exclusive = self._mode == 'overwrite'

    @property
    def constants(self):
        return {'mode': self.mode, 'index_cols': self._index_cols}

    def reset(self):
        super().reset()
        for stream in self.input_streams:
            stream.reset()

    @classmethod
    def transform_function(cls, stream_values, constants):
        from .util.transform import dim
        index_cols = constants['index_cols']
        selection_exprs = [sv['selection_expr'] for sv in stream_values]
        selection_exprs = [expr for expr in selection_exprs if expr is not None]
        selection_expr = None
        if len(selection_exprs) > 0:
            if index_cols:
                if len(selection_exprs) > 1:
                    vals = set.intersection(*(set(expr.ops[2]['args'][0]) for expr in selection_exprs))
                    old = selection_exprs[0]
                    selection_expr = dim('new')
                    selection_expr.dimension = old.dimension
                    selection_expr.ops = list(old.ops)
                    selection_expr.ops[2] = dict(selection_expr.ops[2], args=(list(vals),))
            else:
                selection_expr = selection_exprs[0]
                for expr in selection_exprs[1:]:
                    selection_expr = selection_expr & expr
        return dict(selection_expr=selection_expr)