from contextlib import contextmanager
import unittest
from traits.ctrait import CTrait
from traits.trait_converters import (
from traits.trait_factory import TraitFactory
from traits.trait_handlers import TraitCastType, TraitInstance
from traits.api import Any, Int
def test_trait_cast_trait_factory(self):
    int_trait_factory = TraitFactory(lambda: Int().as_ctrait())
    with reset_trait_factory():
        result = trait_cast(int_trait_factory)
        ct = int_trait_factory.as_ctrait()
    self.assertIsInstance(result, CTrait)
    self.assertIs(result, ct)