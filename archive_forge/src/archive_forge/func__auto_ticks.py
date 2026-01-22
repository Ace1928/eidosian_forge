import warnings
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from . import cm
from .axisgrid import Grid
from ._compat import get_colormap
from .utils import (
def _auto_ticks(self, ax, labels, axis):
    """Determine ticks and ticklabels that minimize overlap."""
    transform = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(transform)
    size = [bbox.width, bbox.height][axis]
    axis = [ax.xaxis, ax.yaxis][axis]
    tick, = axis.set_ticks([0])
    fontsize = tick.label1.get_size()
    max_ticks = int(size // (fontsize / 72))
    if max_ticks < 1:
        return ([], [])
    tick_every = len(labels) // max_ticks + 1
    tick_every = 1 if tick_every == 0 else tick_every
    ticks, labels = self._skip_ticks(labels, tick_every)
    return (ticks, labels)