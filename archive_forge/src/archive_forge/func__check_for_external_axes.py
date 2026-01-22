import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _check_for_external_axes(config):
    """
    Check that all `fig` and `ax` kwargs are either ALL None, 
    or ALL are valid instances of Figures/Axes:
 
    An external Axes object can be passed in three places:
        - mpf.plot() `ax=` kwarg
        - mpf.plot() `volume=` kwarg
        - mpf.make_addplot() `ax=` kwarg
    ALL three places MUST be an Axes object, OR
    ALL three places MUST be None.  But it may not be mixed.
    """
    ap_axlist = []
    addplot = config['addplot']
    if addplot is not None:
        if isinstance(addplot, dict):
            addplot = [addplot]
        elif not _list_of_dict(addplot):
            raise TypeError('addplot must be `dict`, or `list of dict`, NOT ' + str(type(addplot)))
        for apd in addplot:
            ap_axlist.append(apd['ax'])
    if len(ap_axlist) > 0:
        if config['ax'] is None:
            if not all([ax is None for ax in ap_axlist]):
                raise ValueError('make_addplot() `ax` kwarg NOT all None, while plot() `ax` kwarg IS None')
        else:
            if not isinstance(config['ax'], mpl.axes.Axes):
                raise ValueError('plot() ax kwarg must be of type `matplotlib.axis.Axes`')
            if not all([isinstance(ax, mpl.axes.Axes) for ax in ap_axlist]):
                raise ValueError('make_addplot() `ax` kwargs must all be of type `matplotlib.axis.Axes`')
    if config['ax'] is None:
        if isinstance(config['volume'], mpl.axes.Axes):
            raise ValueError('`volume` set to external Axes requires all other Axes be external.')
    elif not isinstance(config['volume'], mpl.axes.Axes) and config['volume'] != False:
        raise ValueError('`volume` must be of type `matplotlib.axis.Axes`')
    external_axes_mode = True if isinstance(config['ax'], mpl.axes.Axes) else False
    return external_axes_mode