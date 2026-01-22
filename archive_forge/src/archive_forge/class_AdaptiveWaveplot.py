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
class AdaptiveWaveplot:
    """A helper class for managing adaptive wave visualizations.

    This object is used to dynamically switch between sample-based and envelope-based
    visualizations of waveforms.
    When the display is zoomed in such that no more than `max_samples` would be
    visible, the sample-based display is used.
    When displaying the raw samples would require more than `max_samples`, an
    envelope-based plot is used instead.

    You should never need to instantiate this object directly, as it is constructed
    automatically by `waveshow`.

    Parameters
    ----------
    times : np.ndarray
        An array containing the time index (in seconds) for each sample.

    y : np.ndarray
        An array containing the (monophonic) wave samples.

    steps : matplotlib.lines.Line2D
        The matplotlib artist used for the sample-based visualization.
        This is constructed by `matplotlib.pyplot.step`.

    envelope : matplotlib.collections.PolyCollection
        The matplotlib artist used for the envelope-based visualization.
        This is constructed by `matplotlib.pyplot.fill_between`.

    sr : number > 0
        The sampling rate of the audio

    max_samples : int > 0
        The maximum number of samples to use for sample-based display.

    transpose : bool
        If `True`, display the wave vertically instead of horizontally.

    See Also
    --------
    waveshow
    """

    def __init__(self, times: np.ndarray, y: np.ndarray, steps: Line2D, envelope: PolyCollection, sr: float=22050, max_samples: int=11025, transpose: bool=False):
        self.times = times
        self.samples = y
        self.steps = steps
        self.envelope = envelope
        self.sr = sr
        self.max_samples = max_samples
        self.transpose = transpose
        self.cid: Optional[int] = None
        self.ax: Optional[mplaxes.Axes] = None

    def __del__(self) -> None:
        """Disconnect callback methods on delete"""
        self.disconnect(strict=True)

    def connect(self, ax: mplaxes.Axes, *, signal: str='xlim_changed') -> None:
        """Connect the adaptor to a signal on an axes object.

        Note that if the adaptor has already been connected to an axes object,
        that connect is first broken and then replaced by a new callback.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to connect with this adaptor's `update`
        signal : string, {"xlim_changed", "ylim_changed"}
            The signal to connect

        See Also
        --------
        disconnect
        """
        self.disconnect()
        self.ax = ax
        self.cid = ax.callbacks.connect(signal, self.update)

    def disconnect(self, *, strict: bool=False) -> None:
        """Disconnect the adaptor's update callback.

        Parameters
        ----------
        strict : bool
            If `True`, remove references to the connected axes.
            If `False` (default), only disconnect the callback.

            This functionality is intended primarily for internal use,
            and should have no observable effects for users.

        See Also
        --------
        connect
        """
        if self.ax:
            self.ax.callbacks.disconnect(self.cid)
            self.cid = None
            if strict:
                self.ax = None

    def update(self, ax: mplaxes.Axes) -> None:
        """Update the matplotlib display according to the current viewport limits.

        This is a callback function, and should not be used directly.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to update
        """
        lims = ax.viewLim
        if self.transpose:
            dim = lims.height * self.sr
            start, end = (lims.y0, lims.y1)
            xdata, ydata = (self.samples, self.times)
            data = self.steps.get_ydata()
        else:
            dim = lims.width * self.sr
            start, end = (lims.x0, lims.x1)
            xdata, ydata = (self.times, self.samples)
            data = self.steps.get_xdata()
        if dim <= self.max_samples:
            self.envelope.set_visible(False)
            self.steps.set_visible(True)
            if start <= data[0] or end >= data[-1]:
                midpoint_time = (start + end) / 2
                idx_start = np.searchsorted(self.times, midpoint_time - 0.5 * self.max_samples / self.sr)
                self.steps.set_data(xdata[idx_start:idx_start + self.max_samples], ydata[idx_start:idx_start + self.max_samples])
        else:
            self.envelope.set_visible(True)
            self.steps.set_visible(False)
        ax.figure.canvas.draw_idle()