import logging
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, collections, cm, colors, contour, ticker
import matplotlib.artist as martist
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
from matplotlib import _docstring
def _add_solids_patches(self, X, Y, C, mappable):
    hatches = mappable.hatches * (len(C) + 1)
    if self._extend_lower():
        hatches = hatches[1:]
    patches = []
    for i in range(len(X) - 1):
        xy = np.array([[X[i, 0], Y[i, 1]], [X[i, 1], Y[i, 0]], [X[i + 1, 1], Y[i + 1, 0]], [X[i + 1, 0], Y[i + 1, 1]]])
        patch = mpatches.PathPatch(mpath.Path(xy), facecolor=self.cmap(self.norm(C[i][0])), hatch=hatches[i], linewidth=0, antialiased=False, alpha=self.alpha)
        self.ax.add_patch(patch)
        patches.append(patch)
    self.solids_patches = patches