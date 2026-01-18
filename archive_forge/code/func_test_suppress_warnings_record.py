import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_suppress_warnings_record():
    sup = suppress_warnings()
    log1 = sup.record()
    with sup:
        log2 = sup.record(message='Some other warning 2')
        sup.filter(message='Some warning')
        warnings.warn('Some warning')
        warnings.warn('Some other warning')
        warnings.warn('Some other warning 2')
        assert_equal(len(sup.log), 2)
        assert_equal(len(log1), 1)
        assert_equal(len(log2), 1)
        assert_equal(log2[0].message.args[0], 'Some other warning 2')
    with sup:
        log2 = sup.record(message='Some other warning 2')
        sup.filter(message='Some warning')
        warnings.warn('Some warning')
        warnings.warn('Some other warning')
        warnings.warn('Some other warning 2')
        assert_equal(len(sup.log), 2)
        assert_equal(len(log1), 1)
        assert_equal(len(log2), 1)
        assert_equal(log2[0].message.args[0], 'Some other warning 2')
    with suppress_warnings() as sup:
        sup.record()
        with suppress_warnings() as sup2:
            sup2.record(message='Some warning')
            warnings.warn('Some warning')
            warnings.warn('Some other warning')
            assert_equal(len(sup2.log), 1)
        assert_equal(len(sup.log), 1)