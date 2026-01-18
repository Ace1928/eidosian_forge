from __future__ import division
import decimal
import math
import re
import struct
import sys
import warnings
from collections import OrderedDict
import numpy as np
from . import Qt, debug, getConfigOption, reload
from .metaarray import MetaArray
from .Qt import QT_LIB, QtCore, QtGui
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def transformToArray(tr):
    """
    Given a QTransform, return a 3x3 numpy array.
    Given a QMatrix4x4, return a 4x4 numpy array.
    
    Example: map an array of x,y coordinates through a transform::
    
        ## coordinates to map are (1,5), (2,6), (3,7), and (4,8)
        coords = np.array([[1,2,3,4], [5,6,7,8], [1,1,1,1]])  # the extra '1' coordinate is needed for translation to work
        
        ## Make an example transform
        tr = QtGui.QTransform()
        tr.translate(3,4)
        tr.scale(2, 0.1)
        
        ## convert to array
        m = pg.transformToArray()[:2]  # ignore the perspective portion of the transformation
        
        ## map coordinates through transform
        mapped = np.dot(m, coords)
    """
    if isinstance(tr, QtGui.QTransform):
        return np.array([[tr.m11(), tr.m21(), tr.m31()], [tr.m12(), tr.m22(), tr.m32()], [tr.m13(), tr.m23(), tr.m33()]])
    elif isinstance(tr, QtGui.QMatrix4x4):
        return np.array(tr.copyDataTo()).reshape(4, 4)
    else:
        raise Exception('Transform argument must be either QTransform or QMatrix4x4.')