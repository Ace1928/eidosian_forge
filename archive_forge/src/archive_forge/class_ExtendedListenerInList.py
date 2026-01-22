import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
class ExtendedListenerInList(HasTraits):
    dummy = Instance(Dummy)
    changed = Bool(False)

    @on_trait_change(['dummy:x'])
    def set_changed(self):
        self.changed = True