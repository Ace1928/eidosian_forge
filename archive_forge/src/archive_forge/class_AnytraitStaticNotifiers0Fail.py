import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, Undefined
class AnytraitStaticNotifiers0Fail(HasTraits):
    fail = Float

    def _anytrait_changed():
        raise Exception('error')