from scipy._lib._util import getfullargspec_no_self as _getfullargspec
import sys
import keyword
import re
import types
import warnings
from itertools import zip_longest
from scipy._lib import doccer
from ._distr_params import distcont, distdiscrete
from scipy._lib._util import check_random_state
from scipy.special import comb, entr
from scipy import optimize
from scipy import integrate
from scipy._lib._finite_differences import _derivative
from scipy import stats
from numpy import (arange, putmask, ones, shape, ndarray, zeros, floor,
import numpy as np
from ._constants import _XMAX, _LOGXMAX
from ._censored_data import CensoredData
from scipy.stats._warnings_errors import FitError
def _construct_default_doc(self, longname=None, docdict=None, discrete='continuous'):
    """Construct instance docstring from the default template."""
    if longname is None:
        longname = 'A'
    self.__doc__ = ''.join([f'{longname} {discrete} random variable.', '\n\n%(before_notes)s\n', docheaders['notes'], '\n%(example)s'])
    self._construct_doc(docdict)