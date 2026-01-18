from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_zero_location():
    """Make sure ``location()`` pays attention to 0-valued args."""
    t = TestTerminal(stream=StringIO(), force_styling=True)
    with t.location(0, 0):
        pass
    eq_(t.stream.getvalue(), unicode_cap('sc') + unicode_parm('cup', 0, 0) + unicode_cap('rc'))