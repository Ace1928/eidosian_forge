import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def test_trait_compound_instance(self):
    """ Test that a deferred Instance() embedded in a TraitCompound handler
        and then a list will not replace the validate method for the outermost
        trait.
        """
    d = Dummy()
    d.xl = [HasTraits()]
    d.x = 'OK'