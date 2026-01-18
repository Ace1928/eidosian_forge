import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_suppress_warnings_forwarding():

    def warn_other_module():

        def warn(arr):
            warnings.warn('Some warning', stacklevel=2)
            return arr
        np.apply_along_axis(warn, 0, [0])
    with suppress_warnings() as sup:
        sup.record()
        with suppress_warnings('always'):
            for i in range(2):
                warnings.warn('Some warning')
        assert_equal(len(sup.log), 2)
    with suppress_warnings() as sup:
        sup.record()
        with suppress_warnings('location'):
            for i in range(2):
                warnings.warn('Some warning')
                warnings.warn('Some warning')
        assert_equal(len(sup.log), 2)
    with suppress_warnings() as sup:
        sup.record()
        with suppress_warnings('module'):
            for i in range(2):
                warnings.warn('Some warning')
                warnings.warn('Some warning')
                warn_other_module()
        assert_equal(len(sup.log), 2)
    with suppress_warnings() as sup:
        sup.record()
        with suppress_warnings('once'):
            for i in range(2):
                warnings.warn('Some warning')
                warnings.warn('Some other warning')
                warn_other_module()
        assert_equal(len(sup.log), 2)