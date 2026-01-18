import json
import click
from cligj import (
import rasterio
import rasterio.crs
from rasterio.rio import options
from rasterio.warp import transform_geom
def update_props(data, **kwds):
    data['properties'].update(**kwds)
    return data