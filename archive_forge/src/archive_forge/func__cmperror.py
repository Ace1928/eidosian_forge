import time as _time
import math as _math
import sys
from operator import index as _index
def _cmperror(x, y):
    raise TypeError("can't compare '%s' to '%s'" % (type(x).__name__, type(y).__name__))