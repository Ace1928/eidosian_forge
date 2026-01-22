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
def _prints_input(self, prefix):
    """
        Return a string representation of node's operands.

        A helper method for `_prints` implementation in derived classes.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
    res = ''
    if hasattr(self, 'input'):
        for i, node in enumerate(self.input):
            if isinstance(node._op, FrameNode):
                res += f'{prefix}input[{i}]: {node._op}\n'
            else:
                res += f'{prefix}input[{i}]:\n' + node._op._prints(prefix + '  ')
    return res