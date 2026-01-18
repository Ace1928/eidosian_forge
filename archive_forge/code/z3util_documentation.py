import common_z3 as CM_Z3
import ctypes
from .z3 import *

    Returned a 'sorted' model (so that it's easier to see)
    The model is sorted by its key,
    e.g. if the model is y = 3 , x = 10, then the result is
    x = 10, y = 3

    EXAMPLES:
    see doctest examples from function prove()

    