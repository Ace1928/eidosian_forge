import sys
import warnings
from functools import partial
from . import _quadpack
import numpy as np
def thefunc(x, *myargs):
    y = -x
    func = myargs[0]
    myargs = (y,) + myargs[1:]
    return -func(*myargs)