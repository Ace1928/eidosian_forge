import datetime
import functools
import logging
from numbers import Real
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.scale as mscale
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.units as munits
@staticmethod
def _translate_tick_params(kw, reverse=False):
    """
        Translate the kwargs supported by `.Axis.set_tick_params` to kwargs
        supported by `.Tick._apply_params`.

        In particular, this maps axis specific names like 'top', 'left'
        to the generic tick1, tick2 logic of the axis. Additionally, there
        are some other name translations.

        Returns a new dict of translated kwargs.

        Note: Use reverse=True to translate from those supported by
        `.Tick._apply_params` back to those supported by
        `.Axis.set_tick_params`.
        """
    kw_ = {**kw}
    allowed_keys = ['size', 'width', 'color', 'tickdir', 'pad', 'labelsize', 'labelcolor', 'labelfontfamily', 'zorder', 'gridOn', 'tick1On', 'tick2On', 'label1On', 'label2On', 'length', 'direction', 'left', 'bottom', 'right', 'top', 'labelleft', 'labelbottom', 'labelright', 'labeltop', 'labelrotation', *_gridline_param_names]
    keymap = {'length': 'size', 'direction': 'tickdir', 'rotation': 'labelrotation', 'left': 'tick1On', 'bottom': 'tick1On', 'right': 'tick2On', 'top': 'tick2On', 'labelleft': 'label1On', 'labelbottom': 'label1On', 'labelright': 'label2On', 'labeltop': 'label2On'}
    if reverse:
        kwtrans = {oldkey: kw_.pop(newkey) for oldkey, newkey in keymap.items() if newkey in kw_}
    else:
        kwtrans = {newkey: kw_.pop(oldkey) for oldkey, newkey in keymap.items() if oldkey in kw_}
    if 'colors' in kw_:
        c = kw_.pop('colors')
        kwtrans['color'] = c
        kwtrans['labelcolor'] = c
    for key in kw_:
        if key not in allowed_keys:
            raise ValueError('keyword %s is not recognized; valid keywords are %s' % (key, allowed_keys))
    kwtrans.update(kw_)
    return kwtrans