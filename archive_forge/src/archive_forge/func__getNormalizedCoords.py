import numpy as np
from .. import functions as fn
from .. import getConfigOption
from .. import Qt
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def _getNormalizedCoords(self):

    def asarray(x):
        if x is None or np.isscalar(x) or isinstance(x, np.ndarray):
            return x
        return np.array(x)
    x = asarray(self.opts.get('x'))
    x0 = asarray(self.opts.get('x0'))
    x1 = asarray(self.opts.get('x1'))
    width = asarray(self.opts.get('width'))
    if x0 is None:
        if width is None:
            raise Exception('must specify either x0 or width')
        if x1 is not None:
            x0 = x1 - width
        elif x is not None:
            x0 = x - width / 2.0
        else:
            raise Exception('must specify at least one of x, x0, or x1')
    if width is None:
        if x1 is None:
            raise Exception('must specify either x1 or width')
        width = x1 - x0
    y = asarray(self.opts.get('y'))
    y0 = asarray(self.opts.get('y0'))
    y1 = asarray(self.opts.get('y1'))
    height = asarray(self.opts.get('height'))
    if y0 is None:
        if height is None:
            y0 = 0
        elif y1 is not None:
            y0 = y1 - height
        elif y is not None:
            y0 = y - height / 2.0
        else:
            y0 = 0
    if height is None:
        if y1 is None:
            raise Exception('must specify either y1 or height')
        height = y1 - y0
    t0, t1 = (x0, x0 + width)
    x0 = np.minimum(t0, t1, dtype=np.float64)
    x1 = np.maximum(t0, t1, dtype=np.float64)
    t0, t1 = (y0, y0 + height)
    y0 = np.minimum(t0, t1, dtype=np.float64)
    y1 = np.maximum(t0, t1, dtype=np.float64)
    return (x0, y0, x1, y1)