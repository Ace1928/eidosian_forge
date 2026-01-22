from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
class Selection:
    """
    Carries out a selection operation on a tables.Table object.

    Parameters
    ----------
    table : a Table object
    where : list of Terms (or convertible to)
    start, stop: indices to start and/or stop selection

    """

    def __init__(self, table: Table, where=None, start: int | None=None, stop: int | None=None) -> None:
        self.table = table
        self.where = where
        self.start = start
        self.stop = stop
        self.condition = None
        self.filter = None
        self.terms = None
        self.coordinates = None
        if is_list_like(where):
            with suppress(ValueError):
                inferred = lib.infer_dtype(where, skipna=False)
                if inferred in ('integer', 'boolean'):
                    where = np.asarray(where)
                    if where.dtype == np.bool_:
                        start, stop = (self.start, self.stop)
                        if start is None:
                            start = 0
                        if stop is None:
                            stop = self.table.nrows
                        self.coordinates = np.arange(start, stop)[where]
                    elif issubclass(where.dtype.type, np.integer):
                        if self.start is not None and (where < self.start).any() or (self.stop is not None and (where >= self.stop).any()):
                            raise ValueError('where must have index locations >= start and < stop')
                        self.coordinates = where
        if self.coordinates is None:
            self.terms = self.generate(where)
            if self.terms is not None:
                self.condition, self.filter = self.terms.evaluate()

    def generate(self, where):
        """where can be a : dict,list,tuple,string"""
        if where is None:
            return None
        q = self.table.queryables()
        try:
            return PyTablesExpr(where, queryables=q, encoding=self.table.encoding)
        except NameError as err:
            qkeys = ','.join(q.keys())
            msg = dedent(f"                The passed where expression: {where}\n                            contains an invalid variable reference\n                            all of the variable references must be a reference to\n                            an axis (e.g. 'index' or 'columns'), or a data_column\n                            The currently defined references are: {qkeys}\n                ")
            raise ValueError(msg) from err

    def select(self):
        """
        generate the selection
        """
        if self.condition is not None:
            return self.table.table.read_where(self.condition.format(), start=self.start, stop=self.stop)
        elif self.coordinates is not None:
            return self.table.table.read_coordinates(self.coordinates)
        return self.table.table.read(start=self.start, stop=self.stop)

    def select_coords(self):
        """
        generate the selection
        """
        start, stop = (self.start, self.stop)
        nrows = self.table.nrows
        if start is None:
            start = 0
        elif start < 0:
            start += nrows
        if stop is None:
            stop = nrows
        elif stop < 0:
            stop += nrows
        if self.condition is not None:
            return self.table.table.get_where_list(self.condition.format(), start=start, stop=stop, sort=True)
        elif self.coordinates is not None:
            return self.coordinates
        return np.arange(start, stop)