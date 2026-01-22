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
def _skip_ticks(self, labels, tickevery):
    """Return ticks and labels at evenly spaced intervals."""
    n = len(labels)
    if tickevery == 0:
        ticks, labels = ([], [])
    elif tickevery == 1:
        ticks, labels = (np.arange(n) + 0.5, labels)
    else:
        start, end, step = (0, n, tickevery)
        ticks = np.arange(start, end, step) + 0.5
        labels = labels[start:end:step]
    return (ticks, labels)