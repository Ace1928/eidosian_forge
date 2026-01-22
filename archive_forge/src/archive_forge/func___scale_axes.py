from __future__ import annotations
from itertools import product
import warnings
import numpy as np
import matplotlib.cm as mcm
import matplotlib.axes as mplaxes
import matplotlib.ticker as mplticker
import matplotlib.pyplot as plt
from . import core
from . import util
from .util.deprecation import rename_kw, Deprecated
from .util.exceptions import ParameterError
from typing import TYPE_CHECKING, Any, Collection, Optional, Union, Callable, Dict
from ._typing import _FloatLike_co
def __scale_axes(axes, ax_type, which):
    """Set the axis scaling"""
    kwargs = dict()
    thresh = 'linthresh'
    base = 'base'
    scale = 'linscale'
    if which == 'x':
        scaler = axes.set_xscale
        limit = axes.set_xlim
    else:
        scaler = axes.set_yscale
        limit = axes.set_ylim
    if ax_type == 'mel':
        mode = 'symlog'
        kwargs[thresh] = 1000.0
        kwargs[base] = 2
    elif ax_type in ['cqt', 'cqt_hz', 'cqt_note', 'cqt_svara', 'vqt_hz', 'vqt_note', 'vqt_fjs']:
        mode = 'log'
        kwargs[base] = 2
    elif ax_type in ['log', 'fft_note', 'fft_svara']:
        mode = 'symlog'
        kwargs[base] = 2
        kwargs[thresh] = float(core.note_to_hz('C2'))
        kwargs[scale] = 0.5
    elif ax_type in ['tempo', 'fourier_tempo']:
        mode = 'log'
        kwargs[base] = 2
        limit(16, 480)
    else:
        return
    scaler(mode, **kwargs)