from traits.api import HasTraits, TraitError
from traits.testing.unittest_tools import UnittestTools
def _create_class(self):
    trait = self.trait

    class Dummy(HasTraits):
        t1 = trait(VALUES)
        t2 = trait(*VALUES)
    return Dummy()