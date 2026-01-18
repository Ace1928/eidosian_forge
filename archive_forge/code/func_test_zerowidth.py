from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_zerowidth(self):
    if sys.version_info >= (3, 7, 0):
        self.assertEqual(regex.split('\\b', 'a b'), ['', 'a', ' ', 'b', ''])
    else:
        self.assertEqual(regex.split('\\b', 'a b'), ['a b'])
    self.assertEqual(regex.split('(?V1)\\b', 'a b'), ['', 'a', ' ', 'b', ''])
    self.assertEqual(regex.findall('^|\\w+', 'foo bar'), ['', 'foo', 'bar'])
    self.assertEqual([m[0] for m in regex.finditer('^|\\w+', 'foo bar')], ['', 'foo', 'bar'])
    self.assertEqual(regex.findall('(?r)^|\\w+', 'foo bar'), ['bar', 'foo', ''])
    self.assertEqual([m[0] for m in regex.finditer('(?r)^|\\w+', 'foo bar')], ['bar', 'foo', ''])
    self.assertEqual(regex.findall('(?V1)^|\\w+', 'foo bar'), ['', 'foo', 'bar'])
    self.assertEqual([m[0] for m in regex.finditer('(?V1)^|\\w+', 'foo bar')], ['', 'foo', 'bar'])
    self.assertEqual(regex.findall('(?rV1)^|\\w+', 'foo bar'), ['bar', 'foo', ''])
    self.assertEqual([m[0] for m in regex.finditer('(?rV1)^|\\w+', 'foo bar')], ['bar', 'foo', ''])
    if sys.version_info >= (3, 7, 0):
        self.assertEqual(regex.split('', 'xaxbxc'), ['', 'x', 'a', 'x', 'b', 'x', 'c', ''])
        self.assertEqual([m for m in regex.splititer('', 'xaxbxc')], ['', 'x', 'a', 'x', 'b', 'x', 'c', ''])
    else:
        self.assertEqual(regex.split('', 'xaxbxc'), ['xaxbxc'])
        self.assertEqual([m for m in regex.splititer('', 'xaxbxc')], ['xaxbxc'])
    if sys.version_info >= (3, 7, 0):
        self.assertEqual(regex.split('(?r)', 'xaxbxc'), ['', 'c', 'x', 'b', 'x', 'a', 'x', ''])
        self.assertEqual([m for m in regex.splititer('(?r)', 'xaxbxc')], ['', 'c', 'x', 'b', 'x', 'a', 'x', ''])
    else:
        self.assertEqual(regex.split('(?r)', 'xaxbxc'), ['xaxbxc'])
        self.assertEqual([m for m in regex.splititer('(?r)', 'xaxbxc')], ['xaxbxc'])
    self.assertEqual(regex.split('(?V1)', 'xaxbxc'), ['', 'x', 'a', 'x', 'b', 'x', 'c', ''])
    self.assertEqual([m for m in regex.splititer('(?V1)', 'xaxbxc')], ['', 'x', 'a', 'x', 'b', 'x', 'c', ''])
    self.assertEqual(regex.split('(?rV1)', 'xaxbxc'), ['', 'c', 'x', 'b', 'x', 'a', 'x', ''])
    self.assertEqual([m for m in regex.splititer('(?rV1)', 'xaxbxc')], ['', 'c', 'x', 'b', 'x', 'a', 'x', ''])