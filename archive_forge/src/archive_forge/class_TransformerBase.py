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
class TransformerBase:
    """Generic GDAL transformer base class

    Notes
    -----
    Subclasses must have a _transformer attribute and implement a `_transform` method.

    """

    def __init__(self):
        self._transformer = None

    @staticmethod
    def _ensure_arr_input(xs, ys, zs=None):
        """Ensure all input coordinates are mapped to array-like objects

        Raises
        ------
        TransformError
            If input coordinates are not all of the same length
        """
        try:
            xs, ys, zs = np.broadcast_arrays(xs, ys, 0 if zs is None else zs)
        except ValueError as error:
            raise TransformError('Input coordinates must be broadcastable to a 1d array') from error
        return (np.atleast_1d(xs), np.atleast_1d(ys), np.atleast_1d(zs))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def rowcol(self, xs, ys, zs=None, op=math.floor, precision=None):
        """Get rows and cols coordinates given geographic coordinates.

        Parameters
        ----------
        xs, ys : float or list of float
            Geographic coordinates
        zs : float or list of float, optional
            Height associated with coordinates. Primarily used for RPC based
            coordinate transformations. Ignored for affine based
            transformations. Default: 0.
        op : function, optional (default: math.floor)
            Function to convert fractional pixels to whole numbers (floor,
            ceiling, round)
        precision : int, optional (default: None)
            This parameter is unused, deprecated in rasterio 1.3.0, and
            will be removed in version 2.0.0.

        Raises
        ------
        ValueError
            If input coordinates are not all equal length

        Returns
        -------
        tuple of float or list of float.

        """
        if precision is not None:
            warnings.warn('The precision parameter is unused, deprecated, and will be removed in 2.0.0.', RasterioDeprecationWarning)
        AS_ARR = True if hasattr(xs, '__iter__') else False
        xs, ys, zs = self._ensure_arr_input(xs, ys, zs=zs)
        try:
            new_cols, new_rows = self._transform(xs, ys, zs, transform_direction=TransformDirection.reverse)
            if len(new_rows) == 1 and (not AS_ARR):
                return (op(new_rows[0]), op(new_cols[0]))
            else:
                return ([op(r) for r in new_rows], [op(c) for c in new_cols])
        except TypeError:
            raise TransformError('Invalid inputs')

    def xy(self, rows, cols, zs=None, offset='center'):
        """
        Returns geographic coordinates given dataset rows and cols coordinates

        Parameters
        ----------
        rows, cols : int or list of int
            Image pixel coordinates
        zs : float or list of float, optional
            Height associated with coordinates. Primarily used for RPC based
            coordinate transformations. Ignored for affine based
            transformations. Default: 0.
        offset : str, optional
            Determines if the returned coordinates are for the center of the
            pixel or for a corner.
        Raises
        ------
        ValueError
            If input coordinates are not all equal length

        Returns
        -------
        tuple of float or list of float

        """
        AS_ARR = True if hasattr(rows, '__iter__') else False
        rows, cols, zs = self._ensure_arr_input(rows, cols, zs=zs)
        if offset == 'center':
            coff, roff = (0.5, 0.5)
        elif offset == 'ul':
            coff, roff = (0, 0)
        elif offset == 'ur':
            coff, roff = (1, 0)
        elif offset == 'll':
            coff, roff = (0, 1)
        elif offset == 'lr':
            coff, roff = (1, 1)
        else:
            raise TransformError('Invalid offset')
        T = IDENTITY.translation(coff, roff)
        offset_rows = []
        offset_cols = []
        try:
            for colrow in zip(cols, rows):
                offset_col, offset_row = T * colrow
                offset_rows.append(offset_row)
                offset_cols.append(offset_col)
            new_xs, new_ys = self._transform(offset_cols, offset_rows, zs, transform_direction=TransformDirection.forward)
            if len(new_xs) == 1 and (not AS_ARR):
                return (new_xs[0], new_ys[0])
            else:
                return (new_xs, new_ys)
        except TypeError:
            raise TransformError('Invalid inputs')

    def _transform(self, xs, ys, zs, transform_direction):
        raise NotImplementedError