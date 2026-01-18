import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_other_getitem_methods_not_called(self):

    class FakeList:

        def __getitem__(inner_self, i):
            self.fail('possibly side-effecting __getitem_ method called')
    self.com.matches(5, 'a[0].', locals_={'a': FakeList()})