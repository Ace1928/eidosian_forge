import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def _start_trajectory(self, xm, ym, broken_streamlines=True):
    """Start recording streamline trajectory"""
    self._traj = []
    self._update_trajectory(xm, ym, broken_streamlines)