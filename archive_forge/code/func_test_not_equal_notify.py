import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._testing import (
from traits.trait_dict_object import TraitDict
from traits.trait_types import Dict, Str
def test_not_equal_notify(self):
    observer1 = DictItemObserver(notify=False, optional=False)
    observer2 = DictItemObserver(notify=True, optional=False)
    self.assertNotEqual(observer1, observer2)