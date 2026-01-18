from traits.api import HasTraits, TraitError
from traits.testing.unittest_tools import UnittestTools
def test_simple_assignment(self):
    dummy = self._create_class()
    with self.assertTraitChanges(dummy, 't1'):
        dummy.t1 = ('other value 1', 77, None)
    with self.assertTraitChanges(dummy, 't2'):
        dummy.t2 = ('other value 2', 99, None)