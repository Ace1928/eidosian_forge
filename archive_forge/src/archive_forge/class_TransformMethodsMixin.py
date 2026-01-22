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
class TransformMethodsMixin:
    """Mixin providing methods for calculations related
    to transforming between rows and columns of the raster
    array and the coordinates.

    These methods are wrappers for the functionality in
    `rasterio.transform` module.

    A subclass with this mixin MUST provide a `transform`
    property.

    """

    def xy(self, row, col, z=None, offset='center', transform_method=TransformMethod.affine, **rpc_options):
        """Get the coordinates x, y of a pixel at row, col.

        The pixel's center is returned by default, but a corner can be returned
        by setting `offset` to one of `ul, ur, ll, lr`.

        Parameters
        ----------
        row : int
            Pixel row.
        col : int
            Pixel column.
        z : float, optional
            Height associated with coordinates. Primarily used for RPC based
            coordinate transformations. Ignored for affine based
            transformations. Default: 0.
        offset : str, optional
            Determines if the returned coordinates are for the center of the
            pixel or for a corner.
        transform_method: TransformMethod, optional
            The coordinate transformation method. Default: `TransformMethod.affine`.
        rpc_options: dict, optional
            Additional arguments passed to GDALCreateRPCTransformer

        Returns
        -------
        tuple
            x, y

        """
        transform = getattr(self, transform_method.value)
        if transform_method is TransformMethod.gcps:
            transform = transform[0]
        if not transform:
            raise AttributeError('Dataset has no {}'.format(transform_method))
        return xy(transform, row, col, zs=z, offset=offset, **rpc_options)

    def index(self, x, y, z=None, op=math.floor, precision=None, transform_method=TransformMethod.affine, **rpc_options):
        """Get the (row, col) index of the pixel containing (x, y).

        Parameters
        ----------
        x : float
            x value in coordinate reference system
        y : float
            y value in coordinate reference system
        z : float, optional
            Height associated with coordinates. Primarily used for RPC based
            coordinate transformations. Ignored for affine based
            transformations. Default: 0.
        op : function, optional (default: math.floor)
            Function to convert fractional pixels to whole numbers (floor,
            ceiling, round)
        transform_method: TransformMethod, optional
            The coordinate transformation method. Default: `TransformMethod.affine`.
        rpc_options: dict, optional
            Additional arguments passed to GDALCreateRPCTransformer
        precision : int, optional
            This parameter is unused, deprecated in rasterio 1.3.0, and
            will be removed in version 2.0.0.

        Returns
        -------
        tuple
            (row index, col index)

        """
        if precision is not None:
            warnings.warn('The precision parameter is unused, deprecated, and will be removed in 2.0.0.', RasterioDeprecationWarning)
        transform = getattr(self, transform_method.value)
        if transform_method is TransformMethod.gcps:
            transform = transform[0]
        if not transform:
            raise AttributeError('Dataset has no {}'.format(transform_method))
        return rowcol(transform, x, y, zs=z, op=op, **rpc_options)