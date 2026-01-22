import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def _set_unmasked_verts(self):
    X = self._coordinates[..., 0]
    Y = self._coordinates[..., 1]
    unmask = self._get_unmasked_polys()
    X1 = np.ma.filled(X[:-1, :-1])[unmask]
    Y1 = np.ma.filled(Y[:-1, :-1])[unmask]
    X2 = np.ma.filled(X[1:, :-1])[unmask]
    Y2 = np.ma.filled(Y[1:, :-1])[unmask]
    X3 = np.ma.filled(X[1:, 1:])[unmask]
    Y3 = np.ma.filled(Y[1:, 1:])[unmask]
    X4 = np.ma.filled(X[:-1, 1:])[unmask]
    Y4 = np.ma.filled(Y[:-1, 1:])[unmask]
    npoly = len(X1)
    xy = np.ma.stack([X1, Y1, X2, Y2, X3, Y3, X4, Y4, X1, Y1], axis=-1)
    verts = xy.reshape((npoly, 5, 2))
    self.set_verts(verts)