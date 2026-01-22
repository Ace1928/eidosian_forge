import functools
import itertools
import logging
import math
from numbers import Integral, Number, Real
import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.category  # Register category unit converter as side effect.
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates  # noqa # Register date unit converter as side effect.
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.units as munits
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import (
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
def _pcolorargs(self, funcname, *args, shading='auto', **kwargs):
    _valid_shading = ['gouraud', 'nearest', 'flat', 'auto']
    try:
        _api.check_in_list(_valid_shading, shading=shading)
    except ValueError:
        _api.warn_external(f"shading value '{shading}' not in list of valid values {_valid_shading}. Setting shading='auto'.")
        shading = 'auto'
    if len(args) == 1:
        C = np.asanyarray(args[0])
        nrows, ncols = C.shape[:2]
        if shading in ['gouraud', 'nearest']:
            X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))
        else:
            X, Y = np.meshgrid(np.arange(ncols + 1), np.arange(nrows + 1))
            shading = 'flat'
        C = cbook.safe_masked_invalid(C, copy=True)
        return (X, Y, C, shading)
    if len(args) == 3:
        C = np.asanyarray(args[2])
        X, Y = args[:2]
        X, Y = self._process_unit_info([('x', X), ('y', Y)], kwargs)
        X, Y = [cbook.safe_masked_invalid(a, copy=True) for a in [X, Y]]
        if funcname == 'pcolormesh':
            if np.ma.is_masked(X) or np.ma.is_masked(Y):
                raise ValueError('x and y arguments to pcolormesh cannot have non-finite values or be of type numpy.ma.MaskedArray with masked values')
        nrows, ncols = C.shape[:2]
    else:
        raise _api.nargs_error(funcname, takes='1 or 3', given=len(args))
    Nx = X.shape[-1]
    Ny = Y.shape[0]
    if X.ndim != 2 or X.shape[0] == 1:
        x = X.reshape(1, Nx)
        X = x.repeat(Ny, axis=0)
    if Y.ndim != 2 or Y.shape[1] == 1:
        y = Y.reshape(Ny, 1)
        Y = y.repeat(Nx, axis=1)
    if X.shape != Y.shape:
        raise TypeError(f'Incompatible X, Y inputs to {funcname}; see help({funcname})')
    if shading == 'auto':
        if ncols == Nx and nrows == Ny:
            shading = 'nearest'
        else:
            shading = 'flat'
    if shading == 'flat':
        if (Nx, Ny) != (ncols + 1, nrows + 1):
            raise TypeError(f"Dimensions of C {C.shape} should be one smaller than X({Nx}) and Y({Ny}) while using shading='flat' see help({funcname})")
    else:
        if (Nx, Ny) != (ncols, nrows):
            raise TypeError('Dimensions of C %s are incompatible with X (%d) and/or Y (%d); see help(%s)' % (C.shape, Nx, Ny, funcname))
        if shading == 'nearest':

            def _interp_grid(X):
                if np.shape(X)[1] > 1:
                    dX = np.diff(X, axis=1) * 0.5
                    if not (np.all(dX >= 0) or np.all(dX <= 0)):
                        _api.warn_external(f'The input coordinates to {funcname} are interpreted as cell centers, but are not monotonically increasing or decreasing. This may lead to incorrectly calculated cell edges, in which case, please supply explicit cell edges to {funcname}.')
                    hstack = np.ma.hstack if np.ma.isMA(X) else np.hstack
                    X = hstack((X[:, [0]] - dX[:, [0]], X[:, :-1] + dX, X[:, [-1]] + dX[:, [-1]]))
                else:
                    X = np.hstack((X, X))
                return X
            if ncols == Nx:
                X = _interp_grid(X)
                Y = _interp_grid(Y)
            if nrows == Ny:
                X = _interp_grid(X.T).T
                Y = _interp_grid(Y.T).T
            shading = 'flat'
    C = cbook.safe_masked_invalid(C, copy=True)
    return (X, Y, C, shading)