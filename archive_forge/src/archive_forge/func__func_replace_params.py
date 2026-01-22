import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
@deprecate_parameter('old1', new_name='new0', start_version='0.10', stop_version='0.12')
@deprecate_parameter('old0', new_name='new1', start_version='0.10', stop_version='0.12')
def _func_replace_params(arg0, old0=DEPRECATED, old1=DEPRECATED, new0=None, new1=None, arg1=None):
    """Expected docstring.

    Parameters
    ----------
    arg0 : int
        First unchanged parameter.
    new0 : int, optional
        First new parameter.

        .. versionadded:: 0.10
    new1 : int, optional
        Second new parameter.

        .. versionadded:: 0.10
    arg1 : int, optional
        Second unchanged parameter.
    """
    return (arg0, old0, old1, new0, new1, arg1)