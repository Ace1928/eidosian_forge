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
def waveshow(y: np.ndarray, *, sr: float=22050, max_points: int=11025, axis: Optional[str]='time', offset: float=0.0, marker: Union[str, MplPath, MarkerStyle]='', where: str='post', label: Optional[str]=None, transpose: bool=False, ax: Optional[mplaxes.Axes]=None, x_axis: Optional[Union[str, Deprecated]]=Deprecated(), **kwargs: Any) -> AdaptiveWaveplot:
    """Visualize a waveform in the time domain.

    This function constructs a plot which adaptively switches between a raw
    samples-based view of the signal (`matplotlib.pyplot.step`) and an
    amplitude-envelope view of the signal (`matplotlib.pyplot.fill_between`)
    depending on the time extent of the plot's viewport.

    More specifically, when the plot spans a time interval of less than ``max_points /
    sr`` (by default, 1/2 second), the samples-based view is used, and otherwise a
    downsampled amplitude envelope is used.
    This is done to limit the complexity of the visual elements to guarantee an
    efficient, visually interpretable plot.

    When using interactive rendering (e.g., in a Jupyter notebook or IPython
    console), the plot will automatically update as the view-port is changed, either
    through widget controls or programmatic updates.

    .. note:: When visualizing stereo waveforms, the amplitude envelope will be generated
        so that the upper limits derive from the left channel, and the lower limits derive
        from the right channel, which can produce a vertically asymmetric plot.

        When zoomed in to the sample view, only the first channel will be shown.
        If you want to visualize both channels at the sample level, it is recommended to
        plot each signal independently.

    Parameters
    ----------
    y : np.ndarray [shape=(n,) or (2,n)]
        audio time series (mono or stereo)

    sr : number > 0 [scalar]
        sampling rate of ``y`` (samples per second)

    max_points : positive integer
        Maximum number of samples to draw.  When the plot covers a time extent
        smaller than ``max_points / sr`` (default: 1/2 second), samples are drawn.

        If drawing raw samples would exceed `max_points`, then a downsampled
        amplitude envelope extracted from non-overlapping windows of `y` is
        visualized instead.  The parameters of the amplitude envelope are defined so
        that the resulting plot cannot produce more than `max_points` frames.

    axis : str or None
        Display style of the axis ticks and tick markers. Accepted values are:

        - 'time' : markers are shown as milliseconds, seconds, minutes, or hours.
                    Values are plotted in units of seconds.

        - 'h' : markers are shown as hours, minutes, and seconds.

        - 'm' : markers are shown as minutes and seconds.

        - 's' : markers are shown as seconds.

        - 'ms' : markers are shown as milliseconds.

        - 'lag' : like time, but past the halfway point counts as negative values.

        - 'lag_h' : same as lag, but in hours.

        - 'lag_m' : same as lag, but in minutes.

        - 'lag_s' : same as lag, but in seconds.

        - 'lag_ms' : same as lag, but in milliseconds.

        - `None`, 'none', or 'off': ticks and tick markers are hidden.

    x_axis : Deprecated
        Equivalent to `axis` parameter, included for backward compatibility.

        .. warning:: This parameter is deprecated as of 0.10.0 and
            will be removed in 1.0.  Use `axis=` instead going forward.

    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.

    offset : float
        Horizontal offset (in seconds) to start the waveform plot

    marker : string
        Marker symbol to use for sample values. (default: no markers)

        See Also: `matplotlib.markers`.

    where : string, {'pre', 'mid', 'post'}
        This setting determines how both waveform and envelope plots interpolate
        between observations.

        See `matplotlib.pyplot.step` for details.

        Default: 'post'

    label : string [optional]
        The label string applied to this plot.
        Note that the label

    transpose : bool
        If `True`, display the wave vertically instead of horizontally.

    **kwargs
        Additional keyword arguments to `matplotlib.pyplot.fill_between` and
        `matplotlib.pyplot.step`.

        Note that only those arguments which are common to both functions will be
        supported.

    Returns
    -------
    librosa.display.AdaptiveWaveplot
        An object of type `librosa.display.AdaptiveWaveplot`

    See Also
    --------
    AdaptiveWaveplot
    matplotlib.pyplot.step
    matplotlib.pyplot.fill_between
    matplotlib.pyplot.fill_betweenx
    matplotlib.markers

    Examples
    --------
    Plot a monophonic waveform with an envelope view

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('choice'), duration=10)
    >>> fig, ax = plt.subplots(nrows=3, sharex=True)
    >>> librosa.display.waveshow(y, sr=sr, ax=ax[0])
    >>> ax[0].set(title='Envelope view, mono')
    >>> ax[0].label_outer()

    Or a stereo waveform

    >>> y, sr = librosa.load(librosa.ex('choice', hq=True), mono=False, duration=10)
    >>> librosa.display.waveshow(y, sr=sr, ax=ax[1])
    >>> ax[1].set(title='Envelope view, stereo')
    >>> ax[1].label_outer()

    Or harmonic and percussive components with transparency

    >>> y, sr = librosa.load(librosa.ex('choice'), duration=10)
    >>> y_harm, y_perc = librosa.effects.hpss(y)
    >>> librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, ax=ax[2], label='Harmonic')
    >>> librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5, ax=ax[2], label='Percussive')
    >>> ax[2].set(title='Multiple waveforms')
    >>> ax[2].legend()

    Zooming in on a plot to show raw sample values

    >>> fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
    >>> ax.set(xlim=[6.0, 6.01], title='Sample view', ylim=[-0.2, 0.2])
    >>> librosa.display.waveshow(y, sr=sr, ax=ax, marker='.', label='Full signal')
    >>> librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, ax=ax2, label='Harmonic')
    >>> librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5, ax=ax2, label='Percussive')
    >>> ax.label_outer()
    >>> ax.legend()
    >>> ax2.legend()

    Plotting a transposed wave along with a self-similarity matrix

    >>> fig, ax = plt.subplot_mosaic("hSSS;hSSS;hSSS;.vvv")
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    >>> sim = librosa.segment.recurrence_matrix(chroma, mode='affinity')
    >>> librosa.display.specshow(sim, ax=ax['S'], sr=sr,
    ...                          x_axis='time', y_axis='time',
    ...                          auto_aspect=False)
    >>> ax['S'].label_outer()
    >>> ax['S'].sharex(ax['v'])
    >>> ax['S'].sharey(ax['h'])
    >>> ax['S'].set(title='Self-similarity')
    >>> librosa.display.waveshow(y, ax=ax['v'])
    >>> ax['v'].label_outer()
    >>> ax['v'].set(title='transpose=False')
    >>> librosa.display.waveshow(y, ax=ax['h'], transpose=True)
    >>> ax['h'].label_outer()
    >>> ax['h'].set(title='transpose=True')
    """
    util.valid_audio(y, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    if max_points <= 0:
        raise ParameterError(f'max_points={max_points} must be strictly positive')
    axes = __check_axes(ax)
    axis = rename_kw(old_name='x_axis', old_value=x_axis, new_name='axis', new_value=axis, version_deprecated='0.10.0', version_removed='1.0')
    if 'color' not in kwargs:
        kwargs.setdefault('color', next(axes._get_lines.prop_cycler)['color'])
    hop_length = max(1, y.shape[-1] // max_points)
    y_env = __envelope(y, hop_length)
    y_bottom, y_top = (-y_env[-1], y_env[0])
    times = offset + core.times_like(y, sr=sr, hop_length=1)
    xdata, ydata = (times[:max_points], y[0, :max_points])
    filler = axes.fill_between
    signal = 'xlim_changed'
    dec_axis = axes.xaxis
    if transpose:
        ydata, xdata = (xdata, ydata)
        filler = axes.fill_betweenx
        signal = 'ylim_changed'
        dec_axis = axes.yaxis
    steps, = axes.step(xdata, ydata, marker=marker, where=where, **kwargs)
    envelope = filler(times[:len(y_top) * hop_length:hop_length], y_bottom, y_top, step=where, label=label, **kwargs)
    adaptor = AdaptiveWaveplot(times, y[0], steps, envelope, sr=sr, max_samples=max_points, transpose=transpose)
    adaptor.connect(axes, signal=signal)
    adaptor.update(axes)
    __decorate_axis(dec_axis, axis)
    return adaptor