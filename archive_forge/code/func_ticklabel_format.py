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
def ticklabel_format(self, *, axis='both', style='', scilimits=None, useOffset=None, useLocale=None, useMathText=None):
    """
        Configure the `.ScalarFormatter` used by default for linear Axes.

        If a parameter is not set, the corresponding property of the formatter
        is left unchanged.

        Parameters
        ----------
        axis : {'x', 'y', 'both'}, default: 'both'
            The axis to configure.  Only major ticks are affected.

        style : {'sci', 'scientific', 'plain'}
            Whether to use scientific notation.
            The formatter default is to use scientific notation.

        scilimits : pair of ints (m, n)
            Scientific notation is used only for numbers outside the range
            10\\ :sup:`m` to 10\\ :sup:`n` (and only if the formatter is
            configured to use scientific notation at all).  Use (0, 0) to
            include all numbers.  Use (m, m) where m != 0 to fix the order of
            magnitude to 10\\ :sup:`m`.
            The formatter default is :rc:`axes.formatter.limits`.

        useOffset : bool or float
            If True, the offset is calculated as needed.
            If False, no offset is used.
            If a numeric value, it sets the offset.
            The formatter default is :rc:`axes.formatter.useoffset`.

        useLocale : bool
            Whether to format the number using the current locale or using the
            C (English) locale.  This affects e.g. the decimal separator.  The
            formatter default is :rc:`axes.formatter.use_locale`.

        useMathText : bool
            Render the offset and scientific notation in mathtext.
            The formatter default is :rc:`axes.formatter.use_mathtext`.

        Raises
        ------
        AttributeError
            If the current formatter is not a `.ScalarFormatter`.
        """
    style = style.lower()
    axis = axis.lower()
    if scilimits is not None:
        try:
            m, n = scilimits
            m + n + 1
        except (ValueError, TypeError) as err:
            raise ValueError('scilimits must be a sequence of 2 integers') from err
    STYLES = {'sci': True, 'scientific': True, 'plain': False, '': None}
    is_sci_style = _api.check_getitem(STYLES, style=style)
    axis_map = {**{k: [v] for k, v in self._axis_map.items()}, 'both': list(self._axis_map.values())}
    axises = _api.check_getitem(axis_map, axis=axis)
    try:
        for axis in axises:
            if is_sci_style is not None:
                axis.major.formatter.set_scientific(is_sci_style)
            if scilimits is not None:
                axis.major.formatter.set_powerlimits(scilimits)
            if useOffset is not None:
                axis.major.formatter.set_useOffset(useOffset)
            if useLocale is not None:
                axis.major.formatter.set_useLocale(useLocale)
            if useMathText is not None:
                axis.major.formatter.set_useMathText(useMathText)
    except AttributeError as err:
        raise AttributeError('This method only works with the ScalarFormatter') from err