from toolz import curried
import uuid
from weakref import WeakValueDictionary
from typing import (
from altair.utils._importers import import_vegafusion
from altair.utils.core import DataFrameLike
from altair.utils.data import DataType, ToValuesReturnType, MaxRowsError
from altair.vegalite.data import default_data_transformer
def using_vegafusion() -> bool:
    """Check whether the vegafusion data transformer is enabled"""
    from altair import data_transformers
    return data_transformers.active == 'vegafusion'