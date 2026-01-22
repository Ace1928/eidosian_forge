import unittest
from traits.api import Array, Bool, HasTraits, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
def _a_changed(self):
    self.event_fired = True