from collections import OrderedDict
import json
import warnings
import click
import rasterio
import rasterio.crs
from rasterio.crs import CRS
from rasterio.dtypes import in_dtype_range
from rasterio.enums import ColorInterp
from rasterio.errors import CRSError
from rasterio.rio import options
from rasterio.transform import guard_transform
def all_handler(ctx, param, value):
    """Get tags from a template file or command line."""
    if ctx.obj and ctx.obj.get('like') and (value is not None):
        ctx.obj['all_like'] = value
        value = ctx.obj.get('like')
    return value