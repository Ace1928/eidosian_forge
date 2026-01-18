from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_overlapped(self):
    self.assertEqual(regex.findall('..', 'abcde'), ['ab', 'cd'])
    self.assertEqual(regex.findall('..', 'abcde', overlapped=True), ['ab', 'bc', 'cd', 'de'])
    self.assertEqual(regex.findall('(?r)..', 'abcde'), ['de', 'bc'])
    self.assertEqual(regex.findall('(?r)..', 'abcde', overlapped=True), ['de', 'cd', 'bc', 'ab'])
    self.assertEqual(regex.findall('(.)(-)(.)', 'a-b-c', overlapped=True), [('a', '-', 'b'), ('b', '-', 'c')])
    self.assertEqual([m[0] for m in regex.finditer('..', 'abcde')], ['ab', 'cd'])
    self.assertEqual([m[0] for m in regex.finditer('..', 'abcde', overlapped=True)], ['ab', 'bc', 'cd', 'de'])
    self.assertEqual([m[0] for m in regex.finditer('(?r)..', 'abcde')], ['de', 'bc'])
    self.assertEqual([m[0] for m in regex.finditer('(?r)..', 'abcde', overlapped=True)], ['de', 'cd', 'bc', 'ab'])
    self.assertEqual([m.groups() for m in regex.finditer('(.)(-)(.)', 'a-b-c', overlapped=True)], [('a', '-', 'b'), ('b', '-', 'c')])
    self.assertEqual([m.groups() for m in regex.finditer('(?r)(.)(-)(.)', 'a-b-c', overlapped=True)], [('b', '-', 'c'), ('a', '-', 'b')])