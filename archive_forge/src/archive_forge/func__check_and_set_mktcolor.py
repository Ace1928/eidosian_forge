import matplotlib.pyplot as plt
import copy
import pprint
import os.path as path
from   mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from   mplfinance._styledata      import _styles
from   mplfinance._helpers        import _mpf_is_color_like
def _check_and_set_mktcolor(candle, **kwarg):
    if len(kwarg) != 1:
        raise ValueError('Expect only ONE kwarg')
    key, value = kwarg.popitem()
    if isinstance(value, dict):
        colors = value
    elif isinstance(value, str) and value == 'inherit'[0:len(value)]:
        colors = candle
    else:
        colors = dict(up=value, down=value)
    for updown in ['up', 'down']:
        if not _mpf_is_color_like(colors[updown]):
            err = f"NOT is_color_like() for {key}['{updown}'] = {colors[updown]}"
            raise ValueError(err)
    return colors