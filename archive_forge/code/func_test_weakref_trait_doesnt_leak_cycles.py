import contextlib
import gc
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Str, WeakRef
from traits.testing.unittest_tools import UnittestTools
def test_weakref_trait_doesnt_leak_cycles(self):
    eggs = Eggs(name='ostrich')
    with restore_gc_state():
        gc.disable()
        gc.collect()
        spam = Spam(eggs=eggs)
        del spam
        self.assertEqual(gc.collect(), 0)