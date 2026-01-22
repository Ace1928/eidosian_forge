from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Literal, Union
import numpy as np
import pandas as pd
from xarray.coding import strings, times, variables
from xarray.coding.variables import SerializationWarning, pop_to
from xarray.core import indexing
from xarray.core.common import (
from xarray.core.utils import emit_user_level_warning
from xarray.core.variable import IndexVariable, Variable
from xarray.namedarray.utils import is_duck_dask_array
def _update_bounds_encoding(variables: T_Variables) -> None:
    """Adds time encoding to time bounds variables.

    Variables handling time bounds ("Cell boundaries" in the CF
    conventions) do not necessarily carry the necessary attributes to be
    decoded. This copies the encoding from the time variable to the
    associated bounds variable so that we write CF-compliant files.

    See Also:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/
         cf-conventions.html#cell-boundaries

    https://github.com/pydata/xarray/issues/2565
    """
    for name, v in variables.items():
        attrs = v.attrs
        encoding = v.encoding
        has_date_units = 'units' in encoding and 'since' in encoding['units']
        is_datetime_type = np.issubdtype(v.dtype, np.datetime64) or contains_cftime_datetimes(v)
        if is_datetime_type and (not has_date_units) and ('bounds' in attrs) and (attrs['bounds'] in variables):
            emit_user_level_warning(f'Variable {name:s} has datetime type and a bounds variable but {name:s}.encoding does not have units specified. The units encodings for {name:s} and {attrs['bounds']} will be determined independently and may not be equal, counter to CF-conventions. If this is a concern, specify a units encoding for {name:s} before writing to a file.')
        if has_date_units and 'bounds' in attrs:
            if attrs['bounds'] in variables:
                bounds_encoding = variables[attrs['bounds']].encoding
                bounds_encoding.setdefault('units', encoding['units'])
                if 'calendar' in encoding:
                    bounds_encoding.setdefault('calendar', encoding['calendar'])