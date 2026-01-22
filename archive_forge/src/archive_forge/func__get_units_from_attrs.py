from __future__ import annotations
import itertools
import textwrap
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from datetime import date, datetime
from inspect import getfullargspec
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
import numpy as np
import pandas as pd
from xarray.core.indexes import PandasMultiIndex
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar, module_available
from xarray.namedarray.pycompat import DuckArrayModule
def _get_units_from_attrs(da: DataArray) -> str:
    """Extracts and formats the unit/units from a attributes."""
    pint_array_type = DuckArrayModule('pint').type
    units = ' [{}]'
    if isinstance(da.data, pint_array_type):
        return units.format(str(da.data.units))
    if 'units' in da.attrs:
        return units.format(da.attrs['units'])
    if 'unit' in da.attrs:
        return units.format(da.attrs['unit'])
    return ''