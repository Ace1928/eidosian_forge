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
def _valid_renko_kwargs():
    """
    Construct and return the "valid renko kwargs table" for the mplfinance.plot(type='renko')
    function. A valid kwargs table is a `dict` of `dict`s. The keys of the outer dict are
    the valid key-words for the function.  The value for each key is a dict containing 3
    specific keys: "Default", "Description" and "Validator" with the following values:
        "Default"      - The default value for the kwarg if none is specified.
        "Description"  - The description for the kwarg.
        "Validator"    - A function that takes the caller specified value for the kwarg,
                         and validates that it is the correct type, and (for kwargs with
                         a limited set of allowed values) may also validate that the
                         kwarg value is one of the allowed values.
    """
    vkwargs = {'brick_size': {'Default': 'atr', 'Description': 'size of each brick on y-axis (typically price).' + ' specify a number, or specify "atr" for average true range.', 'Validator': lambda value: isinstance(value, (float, int)) or value == 'atr'}, 'atr_length': {'Default': 14, 'Description': 'number of periods for atr calculation (if brick size is "atr")', 'Validator': lambda value: isinstance(value, int) or value == 'total'}}
    _validate_vkwargs_dict(vkwargs)
    return vkwargs