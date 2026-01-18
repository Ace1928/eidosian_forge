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
def transform_handler(ctx, param, value):
    """Get transform value from a template file or command line."""
    retval = options.from_like_context(ctx, param, value)
    if retval is None and value:
        try:
            value = json.loads(value)
        except ValueError:
            pass
        try:
            retval = guard_transform(value)
        except Exception:
            raise click.BadParameter("'%s' is not recognized as an Affine array." % value, param=param, param_hint='transform')
    return retval