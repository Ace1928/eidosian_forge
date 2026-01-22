from collections import Counter
import numpy as np
from xarray.coding.times import CFDatetimeCoder, CFTimedeltaCoder
from xarray.conventions import decode_cf
from xarray.core import duck_array_ops
from xarray.core.dataarray import DataArray
from xarray.core.dtypes import get_fill_value
from xarray.namedarray.pycompat import array_type
def _iris_cell_methods_to_str(cell_methods_obj):
    """Converts a Iris cell methods into a string"""
    cell_methods = []
    for cell_method in cell_methods_obj:
        names = ''.join((f'{n}: ' for n in cell_method.coord_names))
        intervals = ' '.join((f'interval: {interval}' for interval in cell_method.intervals))
        comments = ' '.join((f'comment: {comment}' for comment in cell_method.comments))
        extra = ' '.join([intervals, comments]).strip()
        if extra:
            extra = f' ({extra})'
        cell_methods.append(names + cell_method.method + extra)
    return ' '.join(cell_methods)