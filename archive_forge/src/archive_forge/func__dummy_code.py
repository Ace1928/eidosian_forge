from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def _dummy_code(levels):
    return ContrastMatrix(np.eye(len(levels)), _name_levels('', levels))