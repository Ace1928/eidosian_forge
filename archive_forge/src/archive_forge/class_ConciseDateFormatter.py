import datetime
import functools
import logging
import re
from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,
from dateutil.relativedelta import relativedelta
import dateutil.parser
import dateutil.tz
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, ticker, units
class ConciseDateFormatter(ticker.Formatter):
    """
    A `.Formatter` which attempts to figure out the best format to use for the
    date, and to make it as compact as possible, but still be complete. This is
    most useful when used with the `AutoDateLocator`::

    >>> locator = AutoDateLocator()
    >>> formatter = ConciseDateFormatter(locator)

    Parameters
    ----------
    locator : `.ticker.Locator`
        Locator that this axis is using.

    tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
        Ticks timezone, passed to `.dates.num2date`.

    formats : list of 6 strings, optional
        Format strings for 6 levels of tick labelling: mostly years,
        months, days, hours, minutes, and seconds.  Strings use
        the same format codes as `~datetime.datetime.strftime`.  Default is
        ``['%Y', '%b', '%d', '%H:%M', '%H:%M', '%S.%f']``

    zero_formats : list of 6 strings, optional
        Format strings for tick labels that are "zeros" for a given tick
        level.  For instance, if most ticks are months, ticks around 1 Jan 2005
        will be labeled "Dec", "2005", "Feb".  The default is
        ``['', '%Y', '%b', '%b-%d', '%H:%M', '%H:%M']``

    offset_formats : list of 6 strings, optional
        Format strings for the 6 levels that is applied to the "offset"
        string found on the right side of an x-axis, or top of a y-axis.
        Combined with the tick labels this should completely specify the
        date.  The default is::

            ['', '%Y', '%Y-%b', '%Y-%b-%d', '%Y-%b-%d', '%Y-%b-%d %H:%M']

    show_offset : bool, default: True
        Whether to show the offset or not.

    usetex : bool, default: :rc:`text.usetex`
        To enable/disable the use of TeX's math mode for rendering the results
        of the formatter.

    Examples
    --------
    See :doc:`/gallery/ticks/date_concise_formatter`

    .. plot::

        import datetime
        import matplotlib.dates as mdates

        base = datetime.datetime(2005, 2, 1)
        dates = np.array([base + datetime.timedelta(hours=(2 * i))
                          for i in range(732)])
        N = len(dates)
        np.random.seed(19680801)
        y = np.cumsum(np.random.randn(N))

        fig, ax = plt.subplots(constrained_layout=True)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.plot(dates, y)
        ax.set_title('Concise Date Formatter')

    """

    def __init__(self, locator, tz=None, formats=None, offset_formats=None, zero_formats=None, show_offset=True, *, usetex=None):
        """
        Autoformat the date labels.  The default format is used to form an
        initial string, and then redundant elements are removed.
        """
        self._locator = locator
        self._tz = tz
        self.defaultfmt = '%Y'
        if formats:
            if len(formats) != 6:
                raise ValueError('formats argument must be a list of 6 format strings (or None)')
            self.formats = formats
        else:
            self.formats = ['%Y', '%b', '%d', '%H:%M', '%H:%M', '%S.%f']
        if zero_formats:
            if len(zero_formats) != 6:
                raise ValueError('zero_formats argument must be a list of 6 format strings (or None)')
            self.zero_formats = zero_formats
        elif formats:
            self.zero_formats = [''] + self.formats[:-1]
        else:
            self.zero_formats = [''] + self.formats[:-1]
            self.zero_formats[3] = '%b-%d'
        if offset_formats:
            if len(offset_formats) != 6:
                raise ValueError('offset_formats argument must be a list of 6 format strings (or None)')
            self.offset_formats = offset_formats
        else:
            self.offset_formats = ['', '%Y', '%Y-%b', '%Y-%b-%d', '%Y-%b-%d', '%Y-%b-%d %H:%M']
        self.offset_string = ''
        self.show_offset = show_offset
        self._usetex = mpl._val_or_rc(usetex, 'text.usetex')

    def __call__(self, x, pos=None):
        formatter = DateFormatter(self.defaultfmt, self._tz, usetex=self._usetex)
        return formatter(x, pos=pos)

    def format_ticks(self, values):
        tickdatetime = [num2date(value, tz=self._tz) for value in values]
        tickdate = np.array([tdt.timetuple()[:6] for tdt in tickdatetime])
        fmts = self.formats
        zerofmts = self.zero_formats
        offsetfmts = self.offset_formats
        show_offset = self.show_offset
        for level in range(5, -1, -1):
            unique = np.unique(tickdate[:, level])
            if len(unique) > 1:
                if level < 2 and np.any(unique == 1):
                    show_offset = False
                break
            elif level == 0:
                level = 5
        zerovals = [0, 1, 1, 0, 0, 0, 0]
        labels = [''] * len(tickdate)
        for nn in range(len(tickdate)):
            if level < 5:
                if tickdate[nn][level] == zerovals[level]:
                    fmt = zerofmts[level]
                else:
                    fmt = fmts[level]
            elif tickdatetime[nn].second == tickdatetime[nn].microsecond == 0:
                fmt = zerofmts[level]
            else:
                fmt = fmts[level]
            labels[nn] = tickdatetime[nn].strftime(fmt)
        if level >= 5:
            trailing_zeros = min((len(s) - len(s.rstrip('0')) for s in labels if '.' in s), default=None)
            if trailing_zeros:
                for nn in range(len(labels)):
                    if '.' in labels[nn]:
                        labels[nn] = labels[nn][:-trailing_zeros].rstrip('.')
        if show_offset:
            self.offset_string = tickdatetime[-1].strftime(offsetfmts[level])
            if self._usetex:
                self.offset_string = _wrap_in_tex(self.offset_string)
        else:
            self.offset_string = ''
        if self._usetex:
            return [_wrap_in_tex(l) for l in labels]
        else:
            return labels

    def get_offset(self):
        return self.offset_string

    def format_data_short(self, value):
        return num2date(value, tz=self._tz).strftime('%Y-%m-%d %H:%M:%S')