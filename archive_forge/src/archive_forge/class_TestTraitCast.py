from contextlib import contextmanager
import unittest
from traits.ctrait import CTrait
from traits.trait_converters import (
from traits.trait_factory import TraitFactory
from traits.trait_handlers import TraitCastType, TraitInstance
from traits.api import Any, Int
class TestTraitCast(unittest.TestCase):

    def test_trait_cast_ctrait(self):
        ct = Int().as_ctrait()
        result = trait_cast(ct)
        self.assertIs(result, ct)

    def test_trait_cast_trait_type_class(self):
        result = trait_cast(Int)
        self.assertIsInstance(result, CTrait)
        self.assertIsInstance(result.handler, Int)

    def test_trait_cast_trait_type_instance(self):
        trait = Int()
        result = trait_cast(trait)
        self.assertIsInstance(result, CTrait)
        self.assertIs(result.handler, trait)

    def test_trait_cast_trait_factory(self):
        int_trait_factory = TraitFactory(lambda: Int().as_ctrait())
        with reset_trait_factory():
            result = trait_cast(int_trait_factory)
            ct = int_trait_factory.as_ctrait()
        self.assertIsInstance(result, CTrait)
        self.assertIs(result, ct)

    def test_trait_cast_none(self):
        result = trait_cast(None)
        self.assertIsNone(result)

    def test_trait_cast_other(self):
        result = trait_cast(1)
        self.assertIsNone(result)