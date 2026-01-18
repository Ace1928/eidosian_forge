from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_partial(self):
    self.assertEqual(regex.match('ab', 'a', partial=True).partial, True)
    self.assertEqual(regex.match('ab', 'a', partial=True).span(), (0, 1))
    self.assertEqual(regex.match('cats', 'cat', partial=True).partial, True)
    self.assertEqual(regex.match('cats', 'cat', partial=True).span(), (0, 3))
    self.assertEqual(regex.match('cats', 'catch', partial=True), None)
    self.assertEqual(regex.match('abc\\w{3}', 'abcdef', partial=True).partial, False)
    self.assertEqual(regex.match('abc\\w{3}', 'abcdef', partial=True).span(), (0, 6))
    self.assertEqual(regex.match('abc\\w{3}', 'abcde', partial=True).partial, True)
    self.assertEqual(regex.match('abc\\w{3}', 'abcde', partial=True).span(), (0, 5))
    self.assertEqual(regex.match('\\d{4}$', '1234', partial=True).partial, False)
    self.assertEqual(regex.match('\\L<words>', 'post', partial=True, words=['post']).partial, False)
    self.assertEqual(regex.match('\\L<words>', 'post', partial=True, words=['post']).span(), (0, 4))
    self.assertEqual(regex.match('\\L<words>', 'pos', partial=True, words=['post']).partial, True)
    self.assertEqual(regex.match('\\L<words>', 'pos', partial=True, words=['post']).span(), (0, 3))
    self.assertEqual(regex.match('(?fi)\\L<words>', 'POST', partial=True, words=['poﬆ']).partial, False)
    self.assertEqual(regex.match('(?fi)\\L<words>', 'POST', partial=True, words=['poﬆ']).span(), (0, 4))
    self.assertEqual(regex.match('(?fi)\\L<words>', 'POS', partial=True, words=['poﬆ']).partial, True)
    self.assertEqual(regex.match('(?fi)\\L<words>', 'POS', partial=True, words=['poﬆ']).span(), (0, 3))
    self.assertEqual(regex.match('(?fi)\\L<words>', 'poﬆ', partial=True, words=['POS']), None)
    self.assertEqual(regex.match('[a-z]*4R$', 'a', partial=True).span(), (0, 1))
    self.assertEqual(regex.match('[a-z]*4R$', 'ab', partial=True).span(), (0, 2))
    self.assertEqual(regex.match('[a-z]*4R$', 'ab4', partial=True).span(), (0, 3))
    self.assertEqual(regex.match('[a-z]*4R$', 'a4', partial=True).span(), (0, 2))
    self.assertEqual(regex.match('[a-z]*4R$', 'a4R', partial=True).span(), (0, 3))
    self.assertEqual(regex.match('[a-z]*4R$', '4a', partial=True), None)
    self.assertEqual(regex.match('[a-z]*4R$', 'a44', partial=True), None)