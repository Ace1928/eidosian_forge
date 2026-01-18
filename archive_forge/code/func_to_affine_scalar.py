from __future__ import division  # Many analytical derivatives depend on this
from builtins import str, next, map, zip, range, object
import math
from math import sqrt, log, isnan, isinf  # Optimization: no attribute look-up
import re
import sys
import copy
import warnings
import itertools
import inspect
import numbers
import collections
def to_affine_scalar(x):
    """
    Transforms x into a constant affine scalar function
    (AffineScalarFunc), unless it is already an AffineScalarFunc (in
    which case x is returned unchanged).

    Raises an exception unless x belongs to some specific classes of
    objects that are known not to depend on AffineScalarFunc objects
    (which then cannot be considered as constants).
    """
    if isinstance(x, AffineScalarFunc):
        return x
    if isinstance(x, CONSTANT_TYPES):
        return AffineScalarFunc(x, LinearCombination({}))
    raise NotUpcast('%s cannot be converted to a number with uncertainty' % type(x))