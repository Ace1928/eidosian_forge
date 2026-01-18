import contextlib
import gc
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Str, WeakRef
from traits.testing.unittest_tools import UnittestTools
def test_set_and_get(self):
    eggs = Eggs(name='platypus')
    spam = Spam()
    self.assertIsNone(spam.eggs)
    spam.eggs = eggs
    self.assertIs(spam.eggs, eggs)
    del eggs
    self.assertIsNone(spam.eggs)