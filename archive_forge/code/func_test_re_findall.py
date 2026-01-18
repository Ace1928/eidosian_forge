from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_re_findall(self):
    self.assertEqual(regex.findall(':+', 'abc'), [])
    self.assertEqual(regex.findall(':+', 'a:b::c:::d'), [':', '::', ':::'])
    self.assertEqual(regex.findall('(:+)', 'a:b::c:::d'), [':', '::', ':::'])
    self.assertEqual(regex.findall('(:)(:*)', 'a:b::c:::d'), [(':', ''), (':', ':'), (':', '::')])
    self.assertEqual(regex.findall('\\((?P<test>.{0,5}?TEST)\\)', '(MY TEST)'), ['MY TEST'])
    self.assertEqual(regex.findall('\\((?P<test>.{0,3}?TEST)\\)', '(MY TEST)'), ['MY TEST'])
    self.assertEqual(regex.findall('\\((?P<test>.{0,3}?T)\\)', '(MY T)'), ['MY T'])
    self.assertEqual(regex.findall('[^a]{2}[A-Z]', '\n  S'), ['  S'])
    self.assertEqual(regex.findall('[^a]{2,3}[A-Z]', '\n  S'), ['\n  S'])
    self.assertEqual(regex.findall('[^a]{2,3}[A-Z]', '\n   S'), ['   S'])
    self.assertEqual(regex.findall('X(Y[^Y]+?){1,2}( |Q)+DEF', 'XYABCYPPQ\nQ DEF'), [('YPPQ\n', ' ')])
    self.assertEqual(regex.findall('(\\nTest(\\n+.+?){0,2}?)?\\n+End', '\nTest\nxyz\nxyz\nEnd'), [('\nTest\nxyz\nxyz', '\nxyz')])