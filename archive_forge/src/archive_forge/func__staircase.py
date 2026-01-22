import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
@staticmethod
def _staircase(steps):
    return np.concatenate([0.1 * steps[:-1], steps, [10 * steps[1]]])