import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_subclasses_weakref(self):
    """ Make sure that dynamically created subclasses are not held
        strongly by HasTraits.
        """
    previous_subclasses = HasTraits.__subclasses__()
    _create_subclass()
    _create_subclass()
    _create_subclass()
    _create_subclass()
    gc.collect()
    self.assertCountEqual(previous_subclasses, HasTraits.__subclasses__())