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
def _update_data_transforms(self, axisOrder='col-major'):
    """ Sets up the transforms needed to map between input array and display """
    self._dataTransform = QtGui.QTransform()
    self._inverseDataTransform = QtGui.QTransform()
    if self.axisOrder == 'row-major':
        self._dataTransform.scale(1, -1)
        self._dataTransform.rotate(-90)
        self._inverseDataTransform.scale(1, -1)
        self._inverseDataTransform.rotate(-90)