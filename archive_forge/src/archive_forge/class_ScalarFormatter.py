import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
class ScalarFormatter(Formatter):
    """
    Format tick values as a number.

    Parameters
    ----------
    useOffset : bool or float, default: :rc:`axes.formatter.useoffset`
        Whether to use offset notation. See `.set_useOffset`.
    useMathText : bool, default: :rc:`axes.formatter.use_mathtext`
        Whether to use fancy math formatting. See `.set_useMathText`.
    useLocale : bool, default: :rc:`axes.formatter.use_locale`.
        Whether to use locale settings for decimal sign and positive sign.
        See `.set_useLocale`.

    Notes
    -----
    In addition to the parameters above, the formatting of scientific vs.
    floating point representation can be configured via `.set_scientific`
    and `.set_powerlimits`).

    **Offset notation and scientific notation**

    Offset notation and scientific notation look quite similar at first sight.
    Both split some information from the formatted tick values and display it
    at the end of the axis.

    - The scientific notation splits up the order of magnitude, i.e. a
      multiplicative scaling factor, e.g. ``1e6``.

    - The offset notation separates an additive constant, e.g. ``+1e6``. The
      offset notation label is always prefixed with a ``+`` or ``-`` sign
      and is thus distinguishable from the order of magnitude label.

    The following plot with x limits ``1_000_000`` to ``1_000_010`` illustrates
    the different formatting. Note the labels at the right edge of the x axis.

    .. plot::

        lim = (1_000_000, 1_000_010)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'hspace': 2})
        ax1.set(title='offset_notation', xlim=lim)
        ax2.set(title='scientific notation', xlim=lim)
        ax2.xaxis.get_major_formatter().set_useOffset(False)
        ax3.set(title='floating point notation', xlim=lim)
        ax3.xaxis.get_major_formatter().set_useOffset(False)
        ax3.xaxis.get_major_formatter().set_scientific(False)

    """

    def __init__(self, useOffset=None, useMathText=None, useLocale=None):
        if useOffset is None:
            useOffset = mpl.rcParams['axes.formatter.useoffset']
        self._offset_threshold = mpl.rcParams['axes.formatter.offset_threshold']
        self.set_useOffset(useOffset)
        self._usetex = mpl.rcParams['text.usetex']
        self.set_useMathText(useMathText)
        self.orderOfMagnitude = 0
        self.format = ''
        self._scientific = True
        self._powerlimits = mpl.rcParams['axes.formatter.limits']
        self.set_useLocale(useLocale)

    def get_useOffset(self):
        """
        Return whether automatic mode for offset notation is active.

        This returns True if ``set_useOffset(True)``; it returns False if an
        explicit offset was set, e.g. ``set_useOffset(1000)``.

        See Also
        --------
        ScalarFormatter.set_useOffset
        """
        return self._useOffset

    def set_useOffset(self, val):
        """
        Set whether to use offset notation.

        When formatting a set numbers whose value is large compared to their
        range, the formatter can separate an additive constant. This can
        shorten the formatted numbers so that they are less likely to overlap
        when drawn on an axis.

        Parameters
        ----------
        val : bool or float
            - If False, do not use offset notation.
            - If True (=automatic mode), use offset notation if it can make
              the residual numbers significantly shorter. The exact behavior
              is controlled by :rc:`axes.formatter.offset_threshold`.
            - If a number, force an offset of the given value.

        Examples
        --------
        With active offset notation, the values

        ``100_000, 100_002, 100_004, 100_006, 100_008``

        will be formatted as ``0, 2, 4, 6, 8`` plus an offset ``+1e5``, which
        is written to the edge of the axis.
        """
        if val in [True, False]:
            self.offset = 0
            self._useOffset = val
        else:
            self._useOffset = False
            self.offset = val
    useOffset = property(fget=get_useOffset, fset=set_useOffset)

    def get_useLocale(self):
        """
        Return whether locale settings are used for formatting.

        See Also
        --------
        ScalarFormatter.set_useLocale
        """
        return self._useLocale

    def set_useLocale(self, val):
        """
        Set whether to use locale settings for decimal sign and positive sign.

        Parameters
        ----------
        val : bool or None
            *None* resets to :rc:`axes.formatter.use_locale`.
        """
        if val is None:
            self._useLocale = mpl.rcParams['axes.formatter.use_locale']
        else:
            self._useLocale = val
    useLocale = property(fget=get_useLocale, fset=set_useLocale)

    def _format_maybe_minus_and_locale(self, fmt, arg):
        """
        Format *arg* with *fmt*, applying Unicode minus and locale if desired.
        """
        return self.fix_minus((','.join((locale.format_string(part, (arg,), True).replace(',', '{,}') for part in fmt.split(','))) if self._useMathText else locale.format_string(fmt, (arg,), True)) if self._useLocale else fmt % arg)

    def get_useMathText(self):
        """
        Return whether to use fancy math formatting.

        See Also
        --------
        ScalarFormatter.set_useMathText
        """
        return self._useMathText

    def set_useMathText(self, val):
        """
        Set whether to use fancy math formatting.

        If active, scientific notation is formatted as :math:`1.2 \\times 10^3`.

        Parameters
        ----------
        val : bool or None
            *None* resets to :rc:`axes.formatter.use_mathtext`.
        """
        if val is None:
            self._useMathText = mpl.rcParams['axes.formatter.use_mathtext']
            if self._useMathText is False:
                try:
                    from matplotlib import font_manager
                    ufont = font_manager.findfont(font_manager.FontProperties(mpl.rcParams['font.family']), fallback_to_default=False)
                except ValueError:
                    ufont = None
                if ufont == str(cbook._get_data_path('fonts/ttf/cmr10.ttf')):
                    _api.warn_external('cmr10 font should ideally be used with mathtext, set axes.formatter.use_mathtext to True')
        else:
            self._useMathText = val
    useMathText = property(fget=get_useMathText, fset=set_useMathText)

    def __call__(self, x, pos=None):
        """
        Return the format for tick value *x* at position *pos*.
        """
        if len(self.locs) == 0:
            return ''
        else:
            xp = (x - self.offset) / 10.0 ** self.orderOfMagnitude
            if abs(xp) < 1e-08:
                xp = 0
            return self._format_maybe_minus_and_locale(self.format, xp)

    def set_scientific(self, b):
        """
        Turn scientific notation on or off.

        See Also
        --------
        ScalarFormatter.set_powerlimits
        """
        self._scientific = bool(b)

    def set_powerlimits(self, lims):
        """
        Set size thresholds for scientific notation.

        Parameters
        ----------
        lims : (int, int)
            A tuple *(min_exp, max_exp)* containing the powers of 10 that
            determine the switchover threshold. For a number representable as
            :math:`a \\times 10^\\mathrm{exp}` with :math:`1 <= |a| < 10`,
            scientific notation will be used if ``exp <= min_exp`` or
            ``exp >= max_exp``.

            The default limits are controlled by :rc:`axes.formatter.limits`.

            In particular numbers with *exp* equal to the thresholds are
            written in scientific notation.

            Typically, *min_exp* will be negative and *max_exp* will be
            positive.

            For example, ``formatter.set_powerlimits((-3, 4))`` will provide
            the following formatting:
            :math:`1 \\times 10^{-3}, 9.9 \\times 10^{-3}, 0.01,`
            :math:`9999, 1 \\times 10^4`.

        See Also
        --------
        ScalarFormatter.set_scientific
        """
        if len(lims) != 2:
            raise ValueError("'lims' must be a sequence of length 2")
        self._powerlimits = lims

    def format_data_short(self, value):
        if value is np.ma.masked:
            return ''
        if isinstance(value, Integral):
            fmt = '%d'
        else:
            if getattr(self.axis, '__name__', '') in ['xaxis', 'yaxis']:
                if self.axis.__name__ == 'xaxis':
                    axis_trf = self.axis.axes.get_xaxis_transform()
                    axis_inv_trf = axis_trf.inverted()
                    screen_xy = axis_trf.transform((value, 0))
                    neighbor_values = axis_inv_trf.transform(screen_xy + [[-1, 0], [+1, 0]])[:, 0]
                else:
                    axis_trf = self.axis.axes.get_yaxis_transform()
                    axis_inv_trf = axis_trf.inverted()
                    screen_xy = axis_trf.transform((0, value))
                    neighbor_values = axis_inv_trf.transform(screen_xy + [[0, -1], [0, +1]])[:, 1]
                delta = abs(neighbor_values - value).max()
            else:
                a, b = self.axis.get_view_interval()
                delta = (b - a) / 10000.0
            fmt = f'%-#.{cbook._g_sig_digits(value, delta)}g'
        return self._format_maybe_minus_and_locale(fmt, value)

    def format_data(self, value):
        e = math.floor(math.log10(abs(value)))
        s = round(value / 10 ** e, 10)
        significand = self._format_maybe_minus_and_locale('%d' if s % 1 == 0 else '%1.10g', s)
        if e == 0:
            return significand
        exponent = self._format_maybe_minus_and_locale('%d', e)
        if self._useMathText or self._usetex:
            exponent = '10^{%s}' % exponent
            return exponent if s == 1 else f'{significand} \\times {exponent}'
        else:
            return f'{significand}e{exponent}'

    def get_offset(self):
        """
        Return scientific notation, plus offset.
        """
        if len(self.locs) == 0:
            return ''
        if self.orderOfMagnitude or self.offset:
            offsetStr = ''
            sciNotStr = ''
            if self.offset:
                offsetStr = self.format_data(self.offset)
                if self.offset > 0:
                    offsetStr = '+' + offsetStr
            if self.orderOfMagnitude:
                if self._usetex or self._useMathText:
                    sciNotStr = self.format_data(10 ** self.orderOfMagnitude)
                else:
                    sciNotStr = '1e%d' % self.orderOfMagnitude
            if self._useMathText or self._usetex:
                if sciNotStr != '':
                    sciNotStr = '\\times\\mathdefault{%s}' % sciNotStr
                s = f'${sciNotStr}\\mathdefault{{{offsetStr}}}$'
            else:
                s = ''.join((sciNotStr, offsetStr))
            return self.fix_minus(s)
        return ''

    def set_locs(self, locs):
        self.locs = locs
        if len(self.locs) > 0:
            if self._useOffset:
                self._compute_offset()
            self._set_order_of_magnitude()
            self._set_format()

    def _compute_offset(self):
        locs = self.locs
        vmin, vmax = sorted(self.axis.get_view_interval())
        locs = np.asarray(locs)
        locs = locs[(vmin <= locs) & (locs <= vmax)]
        if not len(locs):
            self.offset = 0
            return
        lmin, lmax = (locs.min(), locs.max())
        if lmin == lmax or lmin <= 0 <= lmax:
            self.offset = 0
            return
        abs_min, abs_max = sorted([abs(float(lmin)), abs(float(lmax))])
        sign = math.copysign(1, lmin)
        oom_max = np.ceil(math.log10(abs_max))
        oom = 1 + next((oom for oom in itertools.count(oom_max, -1) if abs_min // 10 ** oom != abs_max // 10 ** oom))
        if (abs_max - abs_min) / 10 ** oom <= 0.01:
            oom = 1 + next((oom for oom in itertools.count(oom_max, -1) if abs_max // 10 ** oom - abs_min // 10 ** oom > 1))
        n = self._offset_threshold - 1
        self.offset = sign * (abs_max // 10 ** oom) * 10 ** oom if abs_max // 10 ** oom >= 10 ** n else 0

    def _set_order_of_magnitude(self):
        if not self._scientific:
            self.orderOfMagnitude = 0
            return
        if self._powerlimits[0] == self._powerlimits[1] != 0:
            self.orderOfMagnitude = self._powerlimits[0]
            return
        vmin, vmax = sorted(self.axis.get_view_interval())
        locs = np.asarray(self.locs)
        locs = locs[(vmin <= locs) & (locs <= vmax)]
        locs = np.abs(locs)
        if not len(locs):
            self.orderOfMagnitude = 0
            return
        if self.offset:
            oom = math.floor(math.log10(vmax - vmin))
        else:
            val = locs.max()
            if val == 0:
                oom = 0
            else:
                oom = math.floor(math.log10(val))
        if oom <= self._powerlimits[0]:
            self.orderOfMagnitude = oom
        elif oom >= self._powerlimits[1]:
            self.orderOfMagnitude = oom
        else:
            self.orderOfMagnitude = 0

    def _set_format(self):
        if len(self.locs) < 2:
            _locs = [*self.locs, *self.axis.get_view_interval()]
        else:
            _locs = self.locs
        locs = (np.asarray(_locs) - self.offset) / 10.0 ** self.orderOfMagnitude
        loc_range = np.ptp(locs)
        if loc_range == 0:
            loc_range = np.max(np.abs(locs))
        if loc_range == 0:
            loc_range = 1
        if len(self.locs) < 2:
            locs = locs[:-2]
        loc_range_oom = int(math.floor(math.log10(loc_range)))
        sigfigs = max(0, 3 - loc_range_oom)
        thresh = 0.001 * 10 ** loc_range_oom
        while sigfigs >= 0:
            if np.abs(locs - np.round(locs, decimals=sigfigs)).max() < thresh:
                sigfigs -= 1
            else:
                break
        sigfigs += 1
        self.format = f'%1.{sigfigs}f'
        if self._usetex or self._useMathText:
            self.format = '$\\mathdefault{%s}$' % self.format