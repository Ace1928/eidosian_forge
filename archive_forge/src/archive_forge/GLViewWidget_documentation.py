from OpenGL.GL import *  # noqa
import OpenGL.GL.framebufferobjects as glfbo  # noqa
from math import cos, radians, sin, tan
import numpy as np
from .. import Vector
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtGui, QtWidgets

        Basic widget for displaying 3D data
          - Rotation/scale controls
          - Axis/grid display
          - Export options

        ================ ==============================================================
        **Arguments:**
        parent           (QObject, optional): Parent QObject. Defaults to None.
        devicePixelRatio No longer in use. High-DPI displays should automatically
                         detect the correct resolution.
        rotationMethod   (str): Mechanism to drive the rotation method, options are
                         'euler' and 'quaternion'. Defaults to 'euler'.
        ================ ==============================================================
        