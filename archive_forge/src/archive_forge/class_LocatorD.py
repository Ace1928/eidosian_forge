import numpy as np
import math
from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
class LocatorD(LocatorBase):

    def __call__(self, v1, v2):
        return select_step360(v1, v2, self.nbins, self._include_last, threshold_factor=1)