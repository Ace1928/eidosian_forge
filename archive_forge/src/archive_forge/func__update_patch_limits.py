from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
def _update_patch_limits(self, patch):
    """Update the data limits for the given patch."""
    if isinstance(patch, mpatches.Rectangle) and (not patch.get_width() and (not patch.get_height())):
        return
    p = patch.get_path()
    vertices = []
    for curve, code in p.iter_bezier(simplify=False):
        _, dzeros = curve.axis_aligned_extrema()
        vertices.append(curve([0, *dzeros, 1]))
    if len(vertices):
        vertices = np.vstack(vertices)
    patch_trf = patch.get_transform()
    updatex, updatey = patch_trf.contains_branch_seperately(self.transData)
    if not (updatex or updatey):
        return
    if self.name != 'rectilinear':
        if updatex and patch_trf == self.get_yaxis_transform():
            updatex = False
        if updatey and patch_trf == self.get_xaxis_transform():
            updatey = False
    trf_to_data = patch_trf - self.transData
    xys = trf_to_data.transform(vertices)
    self.update_datalim(xys, updatex=updatex, updatey=updatey)