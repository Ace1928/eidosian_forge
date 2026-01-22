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
def CIELabColor(L, a, b, alpha=1.0):
    """
    Generates as QColor from CIE L*a*b* values.
    
    Parameters
    ----------
        L: float
            Lightness value ranging from 0 to 100
        a, b: float
            (green/red) and (blue/yellow) coordinates, typically -127 to +127.
        alpha: float, optional
            Opacity, ranging from 0 to 1

    Notes
    -----
    The CIE L*a*b* color space parametrizes color in terms of a luminance `L` 
    and the `a` and `b` coordinates that locate the hue in terms of
    a "green to red" and a "blue to yellow" axis.
    
    These coordinates seek to parametrize human color preception in such a way
    that the Euclidean distance between the coordinates of two colors represents
    the visual difference between these colors. In particular, the difference
    
    ΔE = sqrt( (L1-L2)² + (a1-a2)² + (b1-b2)² ) = 2.3
    
    is considered the smallest "just noticeable difference" between colors.
    
    This simple equation represents the CIE76 standard. Later standards CIE94
    and CIE2000 refine the difference calculation ΔE, while maintaining the 
    L*a*b* coordinates.
    
    Alternative (and arguably more accurate) methods exist to quantify color
    difference, but the CIELab color space remains a convenient approximation.
    
    Under a known illumination, assumed to be white standard illuminant D65 
    here, a CIELab color induces a response in the human eye
    that is described by the tristimulus value XYZ. Once this is known, an
    sRGB color can be calculated to induce the same response.
    
    More information and underlying mathematics can be found in e.g.
    "CIELab Color Space" by Gernot Hoffmann, available at
    http://docs-hoffmann.de/cielab03022003.pdf .
    
    Also see :func:`colorDistance() <pyqtgraph.colorDistance>`.
    """
    vec_XYZ = np.full(3, (L + 16) / 116)
    vec_XYZ[0] += a / 500
    vec_XYZ[2] -= b / 200
    for idx, val in enumerate(vec_XYZ):
        if val > 0.20689:
            vec_XYZ[idx] = vec_XYZ[idx] ** 3
        else:
            vec_XYZ[idx] = (vec_XYZ[idx] - 16 / 116) / 7.787
    vec_XYZ = VECTOR_XYZn * vec_XYZ
    vec_RGB = MATRIX_RGB_FROM_XYZ @ vec_XYZ
    arr_sRGB = np.zeros(3)
    for idx, val in enumerate(vec_RGB[:3]):
        if val > 0.0031308:
            arr_sRGB[idx] = 1.055 * val ** (1 / 2.4) - 0.055
        else:
            arr_sRGB[idx] = 12.92 * val
    arr_sRGB = clip_array(arr_sRGB, 0.0, 1.0)
    return QtGui.QColor.fromRgbF(*arr_sRGB, alpha)