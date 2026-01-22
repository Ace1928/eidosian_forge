import matplotlib.dates  as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.axes   as mpl_axes
import matplotlib.figure as mpl_fig
import pandas as pd
import numpy  as np
import copy
import io
import os
import math
import warnings
import statistics as stat
from itertools import cycle
from mplfinance._utils import _construct_aline_collections
from mplfinance._utils import _construct_hline_collections
from mplfinance._utils import _construct_vline_collections
from mplfinance._utils import _construct_tline_collections
from mplfinance._utils import _construct_mpf_collections
from mplfinance._utils import _construct_pnf_scatter
from mplfinance._widths import _determine_width_config
from mplfinance._utils import _updown_colors
from mplfinance._utils import IntegerIndexDateTimeFormatter
from mplfinance._utils import _mscatter
from mplfinance._utils import _check_and_convert_xlim_configuration
from mplfinance import _styles
from mplfinance._arg_validators import _check_and_prepare_data, _mav_validator, _label_validator
from mplfinance._arg_validators import _get_valid_plot_types, _fill_between_validator
from mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from mplfinance._arg_validators import _kwarg_not_implemented, _bypass_kwarg_validation
from mplfinance._arg_validators import _hlines_validator, _vlines_validator
from mplfinance._arg_validators import _alines_validator, _tlines_validator
from mplfinance._arg_validators import _scale_padding_validator, _yscale_validator
from mplfinance._arg_validators import _valid_panel_id, _check_for_external_axes
from mplfinance._arg_validators import _xlim_validator, _mco_validator, _is_marketcolor_object
from mplfinance._panels import _build_panels
from mplfinance._panels import _set_ticks_on_bottom_panel_only
from mplfinance._helpers import _determine_format_string
from mplfinance._helpers import _list_of_dict
from mplfinance._helpers import _num_or_seq_of_num
from mplfinance._helpers import _adjust_color_brightness
def _addplot_collections(panid, panels, apdict, xdates, config):
    apdata = apdict['data']
    aptype = apdict['type']
    external_axes_mode = apdict['ax'] is not None
    valid_apc_types = ['ohlc', 'candle']
    if aptype not in valid_apc_types:
        raise TypeError('Invalid aptype=' + str(aptype) + '. Must be one of ' + str(valid_apc_types))
    if not isinstance(apdata, pd.DataFrame):
        raise TypeError('addplot type "' + aptype + '" MUST be accompanied by addplot data of type `pd.DataFrame`')
    d, o, h, l, c, v = _check_and_prepare_data(apdata, config)
    mc = apdict['marketcolors']
    if _is_marketcolor_object(mc):
        apstyle = config['style'].copy()
        apstyle['marketcolors'] = mc
    else:
        apstyle = config['style']
    collections = _construct_mpf_collections(aptype, d, xdates, o, h, l, c, v, config, apstyle)
    if not external_axes_mode:
        lo = math.log(max(math.fabs(np.nanmin(l)), 1e-07), 10) - 0.5
        hi = math.log(max(math.fabs(np.nanmax(h)), 1e-07), 10) + 0.5
        secondary_y = _auto_secondary_y(panels, panid, lo, hi)
        if 'auto' != apdict['secondary_y']:
            secondary_y = apdict['secondary_y']
        if secondary_y:
            ax = panels.at[panid, 'axes'][1]
            panels.at[panid, 'used2nd'] = True
        else:
            ax = panels.at[panid, 'axes'][0]
    else:
        ax = apdict['ax']
    for coll in collections:
        ax.add_collection(coll)
    if apdict['mav'] is not None:
        apmavprices = _plot_mav(ax, config, xdates, c, apdict['mav'])
    ax.autoscale_view()
    return ax