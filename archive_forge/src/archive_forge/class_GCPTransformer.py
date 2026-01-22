from contextlib import ExitStack
from functools import partial
import math
import numpy as np
import warnings
from affine import Affine
from rasterio.env import env_ctx_if_needed
from rasterio._transform import (
from rasterio.enums import TransformDirection, TransformMethod
from rasterio.control import GroundControlPoint
from rasterio.rpc import RPC
from rasterio.errors import TransformError, RasterioDeprecationWarning
class GCPTransformer(GCPTransformerBase, GDALTransformerBase):
    """
    Class related to Ground Control Point (GCPs) based
    coordinate transformations.

    Uses GDALCreateGCPTransformer and GDALGCPTransform for computations.
    Ensure that GDAL transformer objects are destroyed by calling `close()`
    method or using context manager interface.

    """

    def __init__(self, gcps):
        if len(gcps) and (not isinstance(gcps[0], GroundControlPoint)):
            raise ValueError('GCPTransformer requires sequence of GroundControlPoint')
        super().__init__(gcps)

    def __repr__(self):
        return '<{} GCPTransformer>'.format(self.closed and 'closed' or 'open')