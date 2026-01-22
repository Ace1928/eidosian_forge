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
class SelectionExprSequence(Derived):
    selection_expr = param.Parameter(default=None, constant=True)
    region_element = param.Parameter(default=None, constant=True)

    def __init__(self, source, mode='overwrite', include_region=True, **params):
        self.mode = mode
        self.include_region = include_region
        sel_expr = SelectionExpr(source, index_cols=params.pop('index_cols'), **params)
        self.history_stream = History(sel_expr)
        input_streams = [self.history_stream]
        super().__init__(source=source, input_streams=input_streams, **params)

    @property
    def constants(self):
        return {'source': self.source, 'mode': self.mode, 'include_region': self.include_region}

    def reset(self):
        self.input_streams[0].clear_history()
        super().reset()

    @classmethod
    def transform_function(cls, stream_values, constants):
        from .core.spaces import DynamicMap
        mode = constants['mode']
        source = constants['source']
        include_region = constants['include_region']
        combined_selection_expr = None
        combined_region_element = None
        for selection_contents in stream_values[0]['values']:
            if selection_contents is None:
                continue
            selection_expr = selection_contents['selection_expr']
            if not selection_expr:
                continue
            region_element = selection_contents['region_element']
            if combined_selection_expr is None or mode == 'overwrite':
                if mode == 'inverse':
                    combined_selection_expr = ~selection_expr
                else:
                    combined_selection_expr = selection_expr
            elif mode == 'intersect':
                combined_selection_expr &= selection_expr
            elif mode == 'union':
                combined_selection_expr |= selection_expr
            else:
                combined_selection_expr &= ~selection_expr
            if isinstance(source, DynamicMap):
                el_type = source.type
            else:
                el_type = source
            combined_region_element = el_type._merge_regions(combined_region_element, region_element, mode)
        return dict(selection_expr=combined_selection_expr, region_element=combined_region_element if include_region else None)