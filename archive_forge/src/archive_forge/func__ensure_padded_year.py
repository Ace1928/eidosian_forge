from __future__ import annotations
import re
import warnings
from collections.abc import Hashable
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Callable, Union
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime, OutOfBoundsTimedelta
from xarray.coding.variables import (
from xarray.core import indexing
from xarray.core.common import contains_cftime_datetimes, is_np_datetime_like
from xarray.core.duck_array_ops import asarray
from xarray.core.formatting import first_n_items, format_timestamp, last_item
from xarray.core.pdcompat import nanosecond_precision_timestamp
from xarray.core.utils import emit_user_level_warning
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import T_ChunkedArray, get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
from xarray.namedarray.utils import is_duck_dask_array
def _ensure_padded_year(ref_date: str) -> str:
    matches_year = re.match('.*\\d{4}.*', ref_date)
    if matches_year:
        return ref_date
    matches_start_digits = re.match('(\\d+)(.*)', ref_date)
    if not matches_start_digits:
        raise ValueError(f'invalid reference date for time units: {ref_date}')
    ref_year, everything_else = (s for s in matches_start_digits.groups())
    ref_date_padded = f'{int(ref_year):04d}{everything_else}'
    warning_msg = f'Ambiguous reference date string: {ref_date}. The first value is assumed to be the year hence will be padded with zeros to remove the ambiguity (the padded reference date string is: {ref_date_padded}). To remove this message, remove the ambiguity by padding your reference date strings with zeros.'
    warnings.warn(warning_msg, SerializationWarning)
    return ref_date_padded