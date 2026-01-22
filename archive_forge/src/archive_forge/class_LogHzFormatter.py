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
class LogHzFormatter(mplticker.Formatter):
    """Ticker formatter for logarithmic frequency

    Parameters
    ----------
    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves

    See Also
    --------
    NoteFormatter
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> fig, ax = plt.subplots(nrows=2)
    >>> ax[0].bar(np.arange(len(values)), values)
    >>> ax[0].yaxis.set_major_formatter(librosa.display.LogHzFormatter())
    >>> ax[0].set(ylabel='Hz')
    >>> ax[1].bar(np.arange(len(values)), values)
    >>> ax[1].yaxis.set_major_formatter(librosa.display.NoteFormatter())
    >>> ax[1].set(ylabel='Note')
    """

    def __init__(self, major: bool=True):
        self.major = major

    def __call__(self, x: float, pos: Optional[int]=None) -> str:
        """Apply the formatter to position"""
        if x <= 0:
            return ''
        vmin, vmax = self.axis.get_view_interval()
        if not self.major and vmax > 4 * max(1, vmin):
            return ''
        return f'{x:g}'