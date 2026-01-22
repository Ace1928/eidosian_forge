import numpy  as np
import pandas as pd
import matplotlib.dates as mdates
import datetime
from itertools import cycle
from matplotlib import colors as mcolors, pyplot as plt
from matplotlib.patches     import Ellipse
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from mplfinance._arg_validators import _alines_validator, _bypass_kwarg_validation
from mplfinance._arg_validators import _xlim_validator, _is_datelike
from mplfinance._styles         import _get_mpfstyle
from mplfinance._helpers        import _mpf_to_rgba
from six.moves import zip
from matplotlib.ticker import Formatter
def _check_and_convert_xlim_configuration(data, config):
    """
    Check, if user entered `xlim` kwarg, if user entered dates
    then we may need to convert them to iloc or matplotlib dates.
    """
    if config['xlim'] is None:
        return None
    xlim = config['xlim']
    if not _xlim_validator(xlim):
        raise ValueError('Bad xlim configuration #1')
    if all([_is_datelike(dt) for dt in xlim]):
        if config['show_nontrading']:
            xlim = [_date_to_mdate(dt) for dt in xlim]
        else:
            xlim = [_date_to_iloc_extrapolate(data.index.to_series(), dt) for dt in xlim]
    return xlim