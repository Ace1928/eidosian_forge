from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
class Treatment(object):
    """Treatment coding (also known as dummy coding).

    This is the default coding.

    For reduced-rank coding, one level is chosen as the "reference", and its
    mean behaviour is represented by the intercept. Each column of the
    resulting matrix represents the difference between the mean of one level
    and this reference level.

    For full-rank coding, classic "dummy" coding is used, and each column of
    the resulting matrix represents the mean of the corresponding level.

    The reference level defaults to the first level, or can be specified
    explicitly.

    .. ipython:: python

       # reduced rank
       dmatrix("C(a, Treatment)", balanced(a=3))
       # full rank
       dmatrix("0 + C(a, Treatment)", balanced(a=3))
       # Setting a reference level
       dmatrix("C(a, Treatment(1))", balanced(a=3))
       dmatrix("C(a, Treatment('a2'))", balanced(a=3))

    Equivalent to R ``contr.treatment``. The R documentation suggests that
    using ``Treatment(reference=-1)`` will produce contrasts that are
    "equivalent to those produced by many (but not all) SAS procedures".
    """

    def __init__(self, reference=None):
        self.reference = reference

    def code_with_intercept(self, levels):
        return _dummy_code(levels)

    def code_without_intercept(self, levels):
        if self.reference is None:
            reference = 0
        else:
            reference = _get_level(levels, self.reference)
        eye = np.eye(len(levels) - 1)
        contrasts = np.vstack((eye[:reference, :], np.zeros((1, len(levels) - 1)), eye[reference:, :]))
        names = _name_levels('T.', levels[:reference] + levels[reference + 1:])
        return ContrastMatrix(contrasts, names)
    __getstate__ = no_pickling