import numpy
from .Qt import QtGui
from . import functions
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions

    Internal function to make an QImage from an ndarray without going
    through the full generality of makeARGB().
    Only certain combinations of input arguments are supported.
    