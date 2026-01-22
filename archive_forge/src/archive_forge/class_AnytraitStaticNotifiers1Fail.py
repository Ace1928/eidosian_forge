import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, Undefined
class AnytraitStaticNotifiers1Fail(HasTraits):
    fail = Float

    def _anytrait_changed(self):
        raise Exception('error')