from __future__ import annotations
import re
import string
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, cast
import numpy as np
import pandas as pd
from dask.dataframe._compat import PANDAS_GE_220, PANDAS_GE_300
from dask.dataframe._pyarrow import is_object_string_dtype
from dask.dataframe.core import tokenize
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.utils import random_state_data
def _with_defaults(_method):
    handler_args, handler_kwargs = default_int_args.get(_method, ((), {}))
    handler_kwargs = handler_kwargs.copy()
    handler_kwargs.update(**kwargs)
    handler_args = args if args else handler_args
    return (handler_args, handler_kwargs)