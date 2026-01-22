import abc
from typing import Generator, Type, Union
import numpy as np
import pandas
import pyarrow as pa
import pyarrow.compute as pc
from pandas.core.dtypes.common import (
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import ColNameCodec, to_arrow_type
def _translate_input(self, mapper):
    """
        Translate the referenced column using `mapper`.

        Parameters
        ----------
        mapper : InputMapper
            A mapper to use for input column translation.

        Returns
        -------
        BaseExpr
            The translated expression.
        """
    return mapper.translate(self)