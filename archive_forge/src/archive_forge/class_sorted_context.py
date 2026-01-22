from itertools import cycle
from operator import itemgetter
import numpy as np
import pandas as pd
import param
from . import util
from .dimension import Dimension, Dimensioned, ViewableElement, asdim
from .util import (
class sorted_context:
    """
    Context manager to temporarily disable sorting on NdMapping
    types. Retains the current sort order, which can be useful as
    an optimization on NdMapping instances where sort=True but the
    items are already known to have been sorted.
    """

    def __init__(self, enabled):
        self.enabled = enabled

    def __enter__(self):
        self._enabled = MultiDimensionalMapping.sort
        MultiDimensionalMapping.sort = self.enabled

    def __exit__(self, exc_type, exc_val, exc_tb):
        MultiDimensionalMapping.sort = self._enabled