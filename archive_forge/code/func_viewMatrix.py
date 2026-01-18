from OpenGL.GL import *  # noqa
import OpenGL.GL.framebufferobjects as glfbo  # noqa
from math import cos, radians, sin, tan
import numpy as np
from .. import Vector
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtGui, QtWidgets
def viewMatrix(self):
    tr = QtGui.QMatrix4x4()
    tr.translate(0.0, 0.0, -self.opts['distance'])
    if self.opts['rotationMethod'] == 'quaternion':
        tr.rotate(self.opts['rotation'])
    else:
        tr.rotate(self.opts['elevation'] - 90, 1, 0, 0)
        tr.rotate(self.opts['azimuth'] + 90, 0, 0, -1)
    center = self.opts['center']
    tr.translate(-center.x(), -center.y(), -center.z())
    return tr