import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_none_returned_when_not_dict_type(self):
    com = autocomplete.DictKeyCompletion()
    local = {'l': ['ab', 'cd']}
    self.assertEqual(com.matches(2, 'l[', locals_=local), None)