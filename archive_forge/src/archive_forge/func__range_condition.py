from itertools import cycle
from operator import itemgetter
import numpy as np
import pandas as pd
import param
from . import util
from .dimension import Dimension, Dimensioned, ViewableElement, asdim
from .util import (
def _range_condition(self, slice):
    if slice.step is None:
        lmbd = lambda x: slice.start <= x < slice.stop
    else:
        lmbd = lambda x: slice.start <= x < slice.stop and (not (x - slice.start) % slice.step)
    return lmbd