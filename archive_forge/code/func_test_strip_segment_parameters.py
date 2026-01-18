import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_strip_segment_parameters(self):
    strip_segment_parameters = urlutils.strip_segment_parameters
    self.assertEqual('/some/path', strip_segment_parameters('/some/path'))
    self.assertEqual('/some/path', strip_segment_parameters('/some/path,tip'))
    self.assertEqual('/some,dir/path', strip_segment_parameters('/some,dir/path,tip'))
    self.assertEqual('/somedir/path', strip_segment_parameters('/somedir/path,heads%2Ftip'))
    self.assertEqual('/somedir/path', strip_segment_parameters('/somedir/path,heads%2Ftip,bar'))
    self.assertEqual('', strip_segment_parameters(',key1=val1'))
    self.assertEqual('foo/', strip_segment_parameters('foo/,key1=val1'))
    self.assertEqual('foo', strip_segment_parameters('foo,key1=val1'))
    self.assertEqual('foo/base,la=bla/other/elements', strip_segment_parameters('foo/base,la=bla/other/elements'))
    self.assertEqual('foo/base,la=bla/other/elements', strip_segment_parameters('foo/base,la=bla/other/elements,a=b'))