import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def _add_trailing_padding(value, padding):
    """Inject the specified number of padding bytes at the end of a dtype"""
    if value.fields is None:
        field_spec = dict(names=['f0'], formats=[value], offsets=[0], itemsize=value.itemsize)
    else:
        fields = value.fields
        names = value.names
        field_spec = dict(names=names, formats=[fields[name][0] for name in names], offsets=[fields[name][1] for name in names], itemsize=value.itemsize)
    field_spec['itemsize'] += padding
    return dtype(field_spec)