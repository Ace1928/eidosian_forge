import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
class AutoLocator(MaxNLocator):
    """
    Dynamically find major tick positions. This is actually a subclass
    of `~matplotlib.ticker.MaxNLocator`, with parameters *nbins = 'auto'*
    and *steps = [1, 2, 2.5, 5, 10]*.
    """

    def __init__(self):
        """
        To know the values of the non-public parameters, please have a
        look to the defaults of `~matplotlib.ticker.MaxNLocator`.
        """
        if mpl.rcParams['_internal.classic_mode']:
            nbins = 9
            steps = [1, 2, 5, 10]
        else:
            nbins = 'auto'
            steps = [1, 2, 2.5, 5, 10]
        super().__init__(nbins=nbins, steps=steps)