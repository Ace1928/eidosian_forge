import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def within_grid(self, xi, yi):
    """Return whether (*xi*, *yi*) is a valid index of the grid."""
    return 0 <= xi <= self.nx - 1 and 0 <= yi <= self.ny - 1