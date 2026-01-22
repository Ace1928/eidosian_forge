from contextlib import contextmanager
import unittest
from traits.ctrait import CTrait
from traits.trait_converters import (
from traits.trait_factory import TraitFactory
from traits.trait_handlers import TraitCastType, TraitInstance
from traits.api import Any, Int
class TestCheckTrait(unittest.TestCase):

    def test_check_trait_ctrait(self):
        ct = Int().as_ctrait()
        result = check_trait(ct)
        self.assertIs(result, ct)

    def test_check_trait_trait_type_class(self):
        result = check_trait(Int)
        self.assertIsInstance(result, CTrait)
        self.assertIsInstance(result.handler, Int)

    def test_check_trait_trait_type_instance(self):
        trait = Int()
        result = check_trait(trait)
        self.assertIsInstance(result, CTrait)
        self.assertIs(result.handler, trait)

    def test_check_trait_trait_factory(self):
        int_trait_factory = TraitFactory(lambda: Int().as_ctrait())
        with reset_trait_factory():
            result = check_trait(int_trait_factory)
            ct = int_trait_factory.as_ctrait()
        self.assertIsInstance(result, CTrait)
        self.assertIs(result, ct)

    def test_check_trait_none(self):
        result = check_trait(None)
        self.assertIsNone(result)

    def test_check_trait_other(self):
        result = check_trait(1)
        self.assertEqual(result, 1)