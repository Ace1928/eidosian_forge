import functools
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy._creation import from_data
from cupy._manipulation import join
class RClass(AxisConcatenator):

    def __init__(self):
        super(RClass, self).__init__()