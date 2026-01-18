import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_normalize_url_hybrid(self):
    normalize_url = urlutils.normalize_url
    eq = self.assertEqual
    eq('file:///foo/', normalize_url('file:///foo/'))
    eq('file:///foo/%20', normalize_url('file:///foo/ '))
    eq('file:///foo/%20', normalize_url('file:///foo/%20'))
    eq('file:///ab_c.d-e/%f:?g&h=i+j;k,L#M$', normalize_url('file:///ab_c.d-e/%f:?g&h=i+j;k,L#M$'))
    eq('http://ab_c.d-e/%f:?g&h=i+j;k,L#M$', normalize_url('http://ab_c.d-e/%f:?g&h=i+j;k,L#M$'))
    eq('http://host/ab/%C2%B5/%C2%B5', normalize_url('http://host/ab/%C2%B5/µ'))
    eq('http://host/~bob%2525-._', normalize_url('http://host/%7Ebob%2525%2D%2E%5F'))
    eq('http://host/~bob%2525-._', normalize_url('http://host/%7Ebob%2525%2D%2E%5F'))