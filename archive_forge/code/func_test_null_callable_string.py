from curses import tigetstr, tparm
from functools import partial
import sys
from nose import SkipTest
from nose.tools import eq_
from six import StringIO
from blessings import *
def test_null_callable_string():
    """Make sure NullCallableString tolerates all numbers and kinds of args it
    might receive."""
    t = TestTerminal(stream=StringIO())
    eq_(t.clear, '')
    eq_(t.move(1, 2), '')
    eq_(t.move_x(1), '')