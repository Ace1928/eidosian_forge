import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._testing import (
from traits.observation._trait_added_observer import (
from traits.trait_types import Str
def test_not_equal_match_func(self):
    observer1 = TraitAddedObserver(match_func=mock.Mock(), optional=True)
    observer2 = TraitAddedObserver(match_func=mock.Mock(), optional=True)
    self.assertNotEqual(observer1, observer2)