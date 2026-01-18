import abc
from typing import TYPE_CHECKING, Dict, List, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.dtypes.common import is_string_dtype
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import EMPTY_ARROW_TABLE, ColNameCodec, get_common_arrow_type
from .db_worker import DbTable
from .expr import InputRefExpr, LiteralExpr, OpExpr
def walk_dfs(self, cb, *args, **kwargs):
    """
        Perform a depth-first walk over a tree.

        Walk over an input in the depth-first order and call a callback function
        for each node.

        Parameters
        ----------
        cb : callable
            A callback function.
        *args : list
            Arguments for the callback.
        **kwargs : dict
            Keyword arguments for the callback.
        """
    if hasattr(self, 'input'):
        for i in self.input:
            i._op.walk_dfs(cb, *args, **kwargs)
    cb(self, *args, **kwargs)