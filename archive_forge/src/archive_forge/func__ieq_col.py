from __future__ import absolute_import, print_function, division
import os
import sys
import pytest
from petl.compat import izip_longest
def _ieq_col(ve, va, re, ra, ir, ic):
    """Print two values when they aren't exactly equals (==)"""
    try:
        eq_(ve, va)
    except AssertionError as ea:
        print('\nrow #%d' % ir, re, ' != ', ra, file=sys.stderr)
        print('col #%d: ' % ic, ve, ' != ', va, file=sys.stderr)
        raise ea