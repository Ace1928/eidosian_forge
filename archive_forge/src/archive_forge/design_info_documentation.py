from __future__ import print_function
import warnings
import numbers
import six
import numpy as np
from patsy import PatsyError
from patsy.util import atleast_2d_column_default
from patsy.compat import OrderedDict
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
from patsy.constraint import linear_constraint
from patsy.contrasts import ContrastMatrix
from patsy.desc import ModelDesc, Term
Create a DesignMatrix, or cast an existing matrix to a DesignMatrix.

        A call like::

          DesignMatrix(my_array)

        will convert an arbitrary array_like object into a DesignMatrix.

        The return from this function is guaranteed to be a two-dimensional
        ndarray with a real-valued floating point dtype, and a
        ``.design_info`` attribute which matches its shape. If the
        `design_info` argument is not given, then one is created via
        :meth:`DesignInfo.from_array` using the given
        `default_column_prefix`.

        Depending on the input array, it is possible this will pass through
        its input unchanged, or create a view.
        