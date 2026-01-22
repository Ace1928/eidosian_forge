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
def _update_bounds_attributes(variables: T_Variables) -> None:
    """Adds time attributes to time bounds variables.

    Variables handling time bounds ("Cell boundaries" in the CF
    conventions) do not necessarily carry the necessary attributes to be
    decoded. This copies the attributes from the time variable to the
    associated boundaries.

    See Also:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/
         cf-conventions.html#cell-boundaries

    https://github.com/pydata/xarray/issues/2565
    """
    for v in variables.values():
        attrs = v.attrs
        units = attrs.get('units')
        has_date_units = isinstance(units, str) and 'since' in units
        if has_date_units and 'bounds' in attrs:
            if attrs['bounds'] in variables:
                bounds_attrs = variables[attrs['bounds']].attrs
                bounds_attrs.setdefault('units', attrs['units'])
                if 'calendar' in attrs:
                    bounds_attrs.setdefault('calendar', attrs['calendar'])