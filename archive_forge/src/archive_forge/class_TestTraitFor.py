from contextlib import contextmanager
import unittest
from traits.ctrait import CTrait
from traits.trait_converters import (
from traits.trait_factory import TraitFactory
from traits.trait_handlers import TraitCastType, TraitInstance
from traits.api import Any, Int
class TestTraitFor(unittest.TestCase):

    def test_trait_for_ctrait(self):
        ct = Int().as_ctrait()
        result = trait_for(ct)
        self.assertIs(result, ct)

    def test_trait_for_trait_type_class(self):
        result = trait_for(Int)
        self.assertIsInstance(result, CTrait)
        self.assertIsInstance(result.handler, Int)

    def test_trait_for_trait_type_instance(self):
        trait = Int()
        result = trait_for(trait)
        self.assertIsInstance(result, CTrait)
        self.assertIs(result.handler, trait)

    def test_trait_for_trait_factory(self):
        int_trait_factory = TraitFactory(lambda: Int().as_ctrait())
        with reset_trait_factory():
            result = trait_for(int_trait_factory)
            ct = int_trait_factory.as_ctrait()
        self.assertIsInstance(result, CTrait)
        self.assertIs(result, ct)

    def test_trait_for_none(self):
        result = trait_for(None)
        self.assertIsInstance(result, CTrait)
        self.assertIsInstance(result.handler, TraitInstance)
        self.assertEqual(result.handler.aClass, type(None))

    def test_trait_for_other(self):
        result = trait_for(1)
        self.assertIsInstance(result, CTrait)
        self.assertIsInstance(result.handler, TraitCastType)
        self.assertEqual(result.handler.aType, int)