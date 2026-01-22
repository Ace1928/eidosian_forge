import math
import textwrap
import sys
import pytest
import threading
import traceback
import time
import numpy as np
from numpy.testing import IS_PYPY
from . import util
class TestGH18335(util.F2PyTest):
    """The reproduction of the reported issue requires specific input that
    extensions may break the issue conditions, so the reproducer is
    implemented as a separate test class. Do not extend this test with
    other tests!
    """
    sources = [util.getpath('tests', 'src', 'callback', 'gh18335.f90')]

    def test_gh18335(self):

        def foo(x):
            x[0] += 1
        r = self.module.gh18335(foo)
        assert r == 123 + 1