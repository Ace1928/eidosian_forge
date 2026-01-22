import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def _format_function(x):
    if np.abs(x) < 1:
        return '.'
    elif np.abs(x) < 2:
        return 'o'
    else:
        return 'O'