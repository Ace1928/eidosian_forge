import numpy as np
import math
from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
class LocatorBase:

    def __init__(self, nbins, include_last=True):
        self.nbins = nbins
        self._include_last = include_last

    def set_params(self, nbins=None):
        if nbins is not None:
            self.nbins = int(nbins)