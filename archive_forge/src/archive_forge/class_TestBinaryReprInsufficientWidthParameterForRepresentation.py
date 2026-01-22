import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys
import numpy as np
from numpy.testing import (
from numpy.core._multiarray_tests import fromstring_null_term_c_api
class TestBinaryReprInsufficientWidthParameterForRepresentation(_DeprecationTestCase):
    """
    If a 'width' parameter is passed into ``binary_repr`` that is insufficient to
    represent the number in base 2 (positive) or 2's complement (negative) form,
    the function used to silently ignore the parameter and return a representation
    using the minimal number of bits needed for the form in question. Such behavior
    is now considered unsafe from a user perspective and will raise an error in the future.
    """

    def test_insufficient_width_positive(self):
        args = (10,)
        kwargs = {'width': 2}
        self.message = 'Insufficient bit width provided. This behavior will raise an error in the future.'
        self.assert_deprecated(np.binary_repr, args=args, kwargs=kwargs)

    def test_insufficient_width_negative(self):
        args = (-5,)
        kwargs = {'width': 2}
        self.message = 'Insufficient bit width provided. This behavior will raise an error in the future.'
        self.assert_deprecated(np.binary_repr, args=args, kwargs=kwargs)