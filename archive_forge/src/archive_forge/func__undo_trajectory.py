import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def _undo_trajectory(self):
    """Remove current trajectory from mask"""
    for t in self._traj:
        self._mask[t] = 0