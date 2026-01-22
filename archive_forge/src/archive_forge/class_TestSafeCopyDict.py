import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
class TestSafeCopyDict(testscenarios.TestWithScenarios):
    scenarios = [('none', {'original': None, 'expected': {}}), ('empty_dict', {'original': {}, 'expected': {}}), ('empty_list', {'original': [], 'expected': {}}), ('dict', {'original': {'a': 1, 'b': 2}, 'expected': {'a': 1, 'b': 2}})]

    def test_expected(self):
        self.assertEqual(self.expected, misc.safe_copy_dict(self.original))
        self.assertFalse(self.expected is misc.safe_copy_dict(self.original))

    def test_mutated_post_copy(self):
        a = {'a': 'b'}
        a_2 = misc.safe_copy_dict(a)
        a['a'] = 'c'
        self.assertEqual('b', a_2['a'])
        self.assertEqual('c', a['a'])