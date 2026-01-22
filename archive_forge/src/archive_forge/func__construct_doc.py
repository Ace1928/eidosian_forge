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
def _construct_doc(self, docdict, shapes_vals=None):
    """Construct the instance docstring with string substitutions."""
    tempdict = docdict.copy()
    tempdict['name'] = self.name or 'distname'
    tempdict['shapes'] = self.shapes or ''
    if shapes_vals is None:
        shapes_vals = ()
    vals = ', '.join(('%.3g' % val for val in shapes_vals))
    tempdict['vals'] = vals
    tempdict['shapes_'] = self.shapes or ''
    if self.shapes and self.numargs == 1:
        tempdict['shapes_'] += ','
    if self.shapes:
        tempdict['set_vals_stmt'] = f'>>> {self.shapes} = {vals}'
    else:
        tempdict['set_vals_stmt'] = ''
    if self.shapes is None:
        for item in ['default', 'before_notes']:
            tempdict[item] = tempdict[item].replace('\n%(shapes)s : array_like\n    shape parameters', '')
    for i in range(2):
        if self.shapes is None:
            self.__doc__ = self.__doc__.replace('%(shapes)s, ', '')
        try:
            self.__doc__ = doccer.docformat(self.__doc__, tempdict)
        except TypeError as e:
            raise Exception(f'Unable to construct docstring for distribution "{self.name}": {repr(e)}') from e
    self.__doc__ = self.__doc__.replace('(, ', '(').replace(', )', ')')