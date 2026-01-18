from contextlib import contextmanager
import unittest
from traits.ctrait import CTrait
from traits.trait_converters import (
from traits.trait_factory import TraitFactory
from traits.trait_handlers import TraitCastType, TraitInstance
from traits.api import Any, Int
def test_trait_from_other(self):
    result = trait_from(1)
    self.assertIsInstance(result, CTrait)
    self.assertIsInstance(result.handler, TraitCastType)
    self.assertEqual(result.handler.aType, int)