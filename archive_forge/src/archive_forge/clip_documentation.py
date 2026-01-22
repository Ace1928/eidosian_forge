import logging
import click
from .helpers import resolve_inout
from . import options
import rasterio
from rasterio.coords import disjoint_bounds
from rasterio.crs import CRS
from rasterio.enums import MaskFlags
from rasterio.windows import Window
Clips a raster using projected or geographic bounds.

    The values of --bounds are presumed to be from the coordinate
    reference system of the input dataset unless the --geographic option
    is used, in which case the values may be longitude and latitude
    bounds. Either JSON, for example "[west, south, east, north]", or
    plain text "west south east north" representations of a bounding box
    are acceptable.

    If using --like, bounds will automatically be transformed to match
    the coordinate reference system of the input.

    Datasets with non-rectilinear geo transforms (i.e. with rotation
    and/or shear) may not be cropped using this command. They must be
    processed with rio-warp.

    Examples
    --------
    $ rio clip input.tif output.tif --bounds xmin ymin xmax ymax

    $ rio clip input.tif output.tif --like template.tif

    