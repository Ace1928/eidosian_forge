import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_set_of_keys_returned_when_matches_found(self):
    com = autocomplete.DictKeyCompletion()
    local = {'d': {'ab': 1, 'cd': 2}}
    self.assertSetEqual(com.matches(2, 'd[', locals_=local), {"'ab']", "'cd']"})