import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_split_segment_parameters(self):
    split_segment_parameters = urlutils.split_segment_parameters
    self.assertEqual(('/some/path', {}), split_segment_parameters('/some/path'))
    self.assertEqual(('/some/path', {'branch': 'tip'}), split_segment_parameters('/some/path,branch=tip'))
    self.assertEqual(('/some,dir/path', {'branch': 'tip'}), split_segment_parameters('/some,dir/path,branch=tip'))
    self.assertEqual(('/somedir/path', {'ref': 'heads%2Ftip'}), split_segment_parameters('/somedir/path,ref=heads%2Ftip'))
    self.assertEqual(('/somedir/path', {'ref': 'heads%2Ftip', 'key1': 'val1'}), split_segment_parameters('/somedir/path,ref=heads%2Ftip,key1=val1'))
    self.assertEqual(('/somedir/path', {'ref': 'heads%2F=tip'}), split_segment_parameters('/somedir/path,ref=heads%2F=tip'))
    self.assertEqual(('', {'key1': 'val1'}), split_segment_parameters(',key1=val1'))
    self.assertEqual(('foo/', {'key1': 'val1'}), split_segment_parameters('foo/,key1=val1'))
    self.assertEqual(('foo/base,key1=val1/other/elements', {}), split_segment_parameters('foo/base,key1=val1/other/elements'))
    self.assertEqual(('foo/base,key1=val1/other/elements', {'key2': 'val2'}), split_segment_parameters('foo/base,key1=val1/other/elements,key2=val2'))
    self.assertRaises(urlutils.InvalidURL, split_segment_parameters, 'foo/base,key1')