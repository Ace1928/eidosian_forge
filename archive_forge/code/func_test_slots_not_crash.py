import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_slots_not_crash(self):
    com = autocomplete.AttrCompletion()
    self.assertSetEqual(com.matches(2, 'A.', locals_={'A': Slots}), {'A.b', 'A.a'})