import sys
import warnings
from functools import partial
from . import _quadpack
import numpy as np
class _RangeFunc:

    def __init__(self, range_):
        self.range_ = range_

    def __call__(self, *args):
        """Return stored value.

        *args needed because range_ can be float or func, and is called with
        variable number of parameters.
        """
        return self.range_