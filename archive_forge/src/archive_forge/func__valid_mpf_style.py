import matplotlib.pyplot as plt
import copy
import pprint
import os.path as path
from   mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from   mplfinance._styledata      import _styles
from   mplfinance._helpers        import _mpf_is_color_like
def _valid_mpf_style(value):
    if value in available_styles():
        return True
    if not isinstance(value, dict):
        return False
    if 'marketcolors' not in value:
        return False
    if not isinstance(value['marketcolors'], dict):
        return False
    for item in ('candle', 'edge', 'wick', 'ohlc', 'volume'):
        if item not in value['marketcolors']:
            return False
        itemcolors = value['marketcolors'][item]
        if not isinstance(itemcolors, dict):
            return False
        if 'up' not in itemcolors or 'down' not in itemcolors:
            return False
    return True