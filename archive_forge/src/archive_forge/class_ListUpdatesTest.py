import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
class ListUpdatesTest(HasTraits):
    a = List
    b = List
    events_received = Int(0)

    @on_trait_change('a[], b[]')
    def _receive_events(self):
        self.events_received += 1