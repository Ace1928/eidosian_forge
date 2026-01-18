from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_re_groupref_exists(self):
    self.assertEqual(regex.match('^(\\()?([^()]+)(?(1)\\))$', '(a)')[:], ('(a)', '(', 'a'))
    self.assertEqual(regex.match('^(\\()?([^()]+)(?(1)\\))$', 'a')[:], ('a', None, 'a'))
    self.assertEqual(regex.match('^(\\()?([^()]+)(?(1)\\))$', 'a)'), None)
    self.assertEqual(regex.match('^(\\()?([^()]+)(?(1)\\))$', '(a'), None)
    self.assertEqual(regex.match('^(?:(a)|c)((?(1)b|d))$', 'ab')[:], ('ab', 'a', 'b'))
    self.assertEqual(regex.match('^(?:(a)|c)((?(1)b|d))$', 'cd')[:], ('cd', None, 'd'))
    self.assertEqual(regex.match('^(?:(a)|c)((?(1)|d))$', 'cd')[:], ('cd', None, 'd'))
    self.assertEqual(regex.match('^(?:(a)|c)((?(1)|d))$', 'a')[:], ('a', 'a', ''))
    p = regex.compile('(?P<g1>a)(?P<g2>b)?((?(g2)c|d))')
    self.assertEqual(p.match('abc')[:], ('abc', 'a', 'b', 'c'))
    self.assertEqual(p.match('ad')[:], ('ad', 'a', None, 'd'))
    self.assertEqual(p.match('abd'), None)
    self.assertEqual(p.match('ac'), None)