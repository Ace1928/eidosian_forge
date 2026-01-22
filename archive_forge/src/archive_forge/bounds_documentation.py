import logging
import os
import click
from cligj import (
from .helpers import write_features, to_lower
from rasterio.rio import options
from rasterio.warp import transform_bounds
Write bounding boxes to stdout as GeoJSON for use with, e.g.,
    geojsonio

      $ rio bounds *.tif | geojsonio

    If a destination crs is passed via dst_crs, it takes precedence over
    the projection parameter.
    