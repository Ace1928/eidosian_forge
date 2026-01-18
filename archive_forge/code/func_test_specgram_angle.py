import contextlib
from collections import namedtuple
import datetime
from decimal import Decimal
from functools import partial
import inspect
import io
from itertools import product
import platform
from types import SimpleNamespace
import dateutil.tz
import numpy as np
from numpy import ma
from cycler import cycler
import pytest
import matplotlib
import matplotlib as mpl
from matplotlib import rc_context, patheffects
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.font_manager as mfont_manager
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.projections.geo import HammerAxes
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import mpl_toolkits.axisartist as AA  # type: ignore
from numpy.testing import (
from matplotlib.testing.decorators import (
@image_comparison(['specgram_angle_freqs.png', 'specgram_phase_freqs.png', 'specgram_angle_noise.png', 'specgram_phase_noise.png'], remove_text=True, tol=0.07, style='default')
def test_specgram_angle():
    """Test axes.specgram in angle and phase modes."""
    matplotlib.rcParams['image.interpolation'] = 'nearest'
    n = 1000
    Fs = 10.0
    fstims = [[Fs / 4, Fs / 5, Fs / 11], [Fs / 4.7, Fs / 5.6, Fs / 11.9]]
    NFFT_freqs = int(10 * Fs / np.min(fstims))
    x = np.arange(0, n, 1 / Fs)
    y = np.sin(2 * np.pi * np.multiply.outer(fstims, x)).sum(axis=1)
    y[:, -1] = 1
    y_freqs = np.hstack(y)
    NFFT_noise = int(10 * Fs / 11)
    np.random.seed(0)
    y_noise = np.concatenate([np.random.standard_normal(n), np.random.rand(n)])
    all_sides = ['default', 'onesided', 'twosided']
    for y, NFFT in [(y_freqs, NFFT_freqs), (y_noise, NFFT_noise)]:
        noverlap = NFFT // 2
        pad_to = int(2 ** np.ceil(np.log2(NFFT)))
        for mode in ['angle', 'phase']:
            for ax, sides in zip(plt.figure().subplots(3), all_sides):
                ax.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to, sides=sides, mode=mode)
                with pytest.raises(ValueError):
                    ax.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to, sides=sides, mode=mode, scale='dB')