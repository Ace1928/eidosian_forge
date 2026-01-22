import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
class NearTestCase(unittest.TestCase):

    def assertNear(a, b, accuracy=1e-05):
        if isinstance(a, (float, int)):
            if abs(a - b) > accuracy:
                raise AssertionError('%s not near %s' % (a, b))
        else:
            for ae, be in zip(a, b):
                if abs(ae - be) > accuracy:
                    raise AssertionError('%s not near %s' % (a, b))
    assertNear = staticmethod(assertNear)