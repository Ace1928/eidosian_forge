from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
class Helmert(object):
    """Helmert contrasts.

    Compares the second level with the first, the third with the average of
    the first two, and so on.

    For full-rank coding, a standard intercept term is added.

    .. warning:: There are multiple definitions of 'Helmert coding' in
       use. Make sure this is the one you expect before trying to interpret
       your results!

    Examples:

    .. ipython:: python

       # Reduced rank
       dmatrix("C(a, Helmert)", balanced(a=4))
       # Full rank
       dmatrix("0 + C(a, Helmert)", balanced(a=4))

    This is equivalent to R's `contr.helmert`.
    """

    def _helmert_contrast(self, levels):
        n = len(levels)
        contr = np.zeros((n, n - 1))
        contr[1:][np.diag_indices(n - 1)] = np.arange(1, n)
        contr[np.triu_indices(n - 1)] = -1
        return contr

    def code_with_intercept(self, levels):
        contrast = np.column_stack((np.ones(len(levels)), self._helmert_contrast(levels)))
        column_suffixes = _name_levels('H.', ['intercept'] + list(levels[1:]))
        return ContrastMatrix(contrast, column_suffixes)

    def code_without_intercept(self, levels):
        contrast = self._helmert_contrast(levels)
        return ContrastMatrix(contrast, _name_levels('H.', levels[1:]))
    __getstate__ = no_pickling