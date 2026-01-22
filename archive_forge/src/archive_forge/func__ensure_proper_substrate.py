import warnings
from collections.abc import Callable
import numpy
from .. import colormap
from .. import debug as debug
from .. import functions as fn
from .. import functions_qimage
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..util.cupy_helper import getCupy
from .GraphicsObject import GraphicsObject
@staticmethod
def _ensure_proper_substrate(data, substrate):
    if data is None or isinstance(data, Callable) or isinstance(data, substrate.ndarray):
        return data
    cupy = getCupy()
    if substrate == cupy and (not isinstance(data, cupy.ndarray)):
        data = cupy.asarray(data)
    elif substrate == numpy:
        if cupy is not None and isinstance(data, cupy.ndarray):
            data = data.get()
        else:
            data = numpy.asarray(data)
    return data