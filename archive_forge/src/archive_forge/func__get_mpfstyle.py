import matplotlib.pyplot as plt
import copy
import pprint
import os.path as path
from   mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from   mplfinance._styledata      import _styles
from   mplfinance._helpers        import _mpf_is_color_like
def _get_mpfstyle(style):
    """ 
    Return a copy of the specified pre-defined mpfstyle.  We return
    a copy, because returning the original will effectively return 
    a pointer which allows style's definition to be modified.
    """
    return copy.deepcopy(_styles[style])