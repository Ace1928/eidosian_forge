import json
import logging
from math import ceil
import os
from affine import Affine
import click
import cligj
import rasterio
from rasterio.errors import CRSError
from rasterio.coords import disjoint_bounds
from rasterio.rio import options
from rasterio.rio.helpers import resolve_inout
import rasterio.shutil
Rasterize GeoJSON into a new or existing raster.

    If the output raster exists, rio-rasterize will rasterize feature
    values into all bands of that raster.  The GeoJSON is assumed to be
    in the same coordinate reference system as the output unless
    --src-crs is provided.

    --default_value or property values when using --property must be
    using a data type valid for the data type of that raster.

    If a template raster is provided using the --like option, the affine
    transform and data type from that raster will be used to create the
    output.  Only a single band will be output.

    The GeoJSON is assumed to be in the same coordinate reference system
    unless --src-crs is provided.

    --default_value or property values when using --property must be
    using a data type valid for the data type of that raster.

    --driver, --bounds, --dimensions, --res, --nodata are ignored when
    output exists or --like raster is provided

    If the output does not exist and --like raster is not provided, the
    input GeoJSON will be used to determine the bounds of the output
    unless provided using --bounds.

    --dimensions or --res are required in this case.

    If --res is provided, the bottom and right coordinates of bounds are
    ignored.

    Note
    ----

    The GeoJSON is not projected to match the coordinate reference
    system of the output or --like rasters at this time.  This
    functionality may be added in the future.

    