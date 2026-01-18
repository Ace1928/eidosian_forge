import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._testing import (
from traits.trait_dict_object import TraitDict
from traits.trait_types import Dict, Str
def test_not_a_dict(self):
    observer = create_observer(optional=False)
    with self.assertRaises(ValueError) as exception_context:
        list(observer.iter_observables(None))
    self.assertIn('Expected a TraitDict to be observed, got', str(exception_context.exception))