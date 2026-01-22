import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import na_values
from rpy2.rinterface import IntSexpVector
from rpy2.rinterface import ListSexpVector
from rpy2.rinterface import SexpVector
from rpy2.rinterface import StrSexpVector
import datetime
import functools
import math
import numpy  # type: ignore
import pandas  # type: ignore
import pandas.core.series  # type: ignore
from pandas.core.frame import DataFrame as PandasDataFrame  # type: ignore
from pandas.core.dtypes.api import is_datetime64_any_dtype  # type: ignore
import warnings
from collections import OrderedDict
from rpy2.robjects.vectors import (BoolVector,
import rpy2.robjects.numpy2ri as numpy2ri
def _flatten_dataframe(obj, colnames_lst):
    """Make each element in a list of columns or group of
    columns iterable.

    This is an helper function to make the "flattening" of columns
    in an R data frame easier as each item in the top-level iterable
    can be a column or list or records (themselves each with an
    arbitrary number of columns).

    Args:
    - colnames_list: an *empty* list that will be populated with
    column names.
    """
    for i, n in enumerate(obj.colnames):
        col = obj[i]
        if isinstance(col, ListSexpVector):
            _ = _records_to_columns(col)
            colnames_lst.extend(((n, subn) for subn in _.keys()))
            for subcol in _.values():
                yield subcol
        else:
            colnames_lst.append(n)
            yield col