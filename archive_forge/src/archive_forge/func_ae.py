import math
import pytest
from mpmath import *
def ae(x, y, tol=1e-12):
    return abs(x - y) <= abs(y) * tol