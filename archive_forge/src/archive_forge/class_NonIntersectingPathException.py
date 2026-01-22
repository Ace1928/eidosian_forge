from functools import lru_cache
import math
import warnings
import numpy as np
from matplotlib import _api
class NonIntersectingPathException(ValueError):
    pass