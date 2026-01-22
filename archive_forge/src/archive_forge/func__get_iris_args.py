from collections import Counter
import numpy as np
from xarray.coding.times import CFDatetimeCoder, CFTimedeltaCoder
from xarray.conventions import decode_cf
from xarray.core import duck_array_ops
from xarray.core.dataarray import DataArray
from xarray.core.dtypes import get_fill_value
from xarray.namedarray.pycompat import array_type
def _get_iris_args(attrs):
    """Converts the xarray attrs into args that can be passed into Iris"""
    import cf_units
    args = {'attributes': _filter_attrs(attrs, iris_forbidden_keys)}
    args.update(_pick_attrs(attrs, ('standard_name', 'long_name')))
    unit_args = _pick_attrs(attrs, ('calendar',))
    if 'units' in attrs:
        args['units'] = cf_units.Unit(attrs['units'], **unit_args)
    return args