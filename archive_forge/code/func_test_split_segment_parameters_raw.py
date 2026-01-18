import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_split_segment_parameters_raw(self):
    split_segment_parameters_raw = urlutils.split_segment_parameters_raw
    self.assertEqual(('/some/path', []), split_segment_parameters_raw('/some/path'))
    self.assertEqual(('/some/path', ['tip']), split_segment_parameters_raw('/some/path,tip'))
    self.assertEqual(('/some,dir/path', ['tip']), split_segment_parameters_raw('/some,dir/path,tip'))
    self.assertEqual(('/somedir/path', ['heads%2Ftip']), split_segment_parameters_raw('/somedir/path,heads%2Ftip'))
    self.assertEqual(('/somedir/path', ['heads%2Ftip', 'bar']), split_segment_parameters_raw('/somedir/path,heads%2Ftip,bar'))
    self.assertEqual(('', ['key1=val1']), split_segment_parameters_raw(',key1=val1'))
    self.assertEqual(('foo/', ['key1=val1']), split_segment_parameters_raw('foo/,key1=val1'))
    self.assertEqual(('foo', ['key1=val1']), split_segment_parameters_raw('foo,key1=val1'))
    self.assertEqual(('foo/base,la=bla/other/elements', []), split_segment_parameters_raw('foo/base,la=bla/other/elements'))
    self.assertEqual(('foo/base,la=bla/other/elements', ['a=b']), split_segment_parameters_raw('foo/base,la=bla/other/elements,a=b'))