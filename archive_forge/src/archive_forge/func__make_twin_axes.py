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
def _make_twin_axes(self, *args, **kwargs):
    """Make a twinx Axes of self. This is used for twinx and twiny."""
    if 'sharex' in kwargs and 'sharey' in kwargs:
        if kwargs['sharex'] is not self and kwargs['sharey'] is not self:
            raise ValueError('Twinned Axes may share only one axis')
    ss = self.get_subplotspec()
    if ss:
        twin = self.figure.add_subplot(ss, *args, **kwargs)
    else:
        twin = self.figure.add_axes(self.get_position(True), *args, **kwargs, axes_locator=_TransformedBoundsLocator([0, 0, 1, 1], self.transAxes))
    self.set_adjustable('datalim')
    twin.set_adjustable('datalim')
    self._twinned_axes.join(self, twin)
    return twin