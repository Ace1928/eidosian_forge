import json
import logging
import click
from .helpers import resolve_inout
from . import options
import rasterio
import rasterio.shutil
from rasterio.mask import mask as mask_tool
Masks in raster using GeoJSON features (masks out all areas not covered
    by features), and optionally crops the output raster to the extent of the
    features.  Features are assumed to be in the same coordinate reference
    system as the input raster.

    GeoJSON must be the first input file or provided from stdin:

    > rio mask input.tif output.tif --geojson-mask features.json

    > rio mask input.tif output.tif --geojson-mask - < features.json

    If the output raster exists, it will be completely overwritten with the
    results of this operation.

    The result is always equal to or within the bounds of the input raster.

    --crop and --invert options are mutually exclusive.

    --crop option is not valid if features are completely outside extent of
    input raster.
    