import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._set_item_observer import SetItemObserver
from traits.observation._testing import (
from traits.trait_set_object import TraitSet
from traits.trait_types import Set
class TestSetItemObserverEqualHash(unittest.TestCase):
    """ Test SetItemObserver __eq__, __hash__ and immutability. """

    def test_not_equal_notify(self):
        observer1 = SetItemObserver(notify=False, optional=False)
        observer2 = SetItemObserver(notify=True, optional=False)
        self.assertNotEqual(observer1, observer2)

    def test_not_equal_optional(self):
        observer1 = SetItemObserver(notify=True, optional=True)
        observer2 = SetItemObserver(notify=True, optional=False)
        self.assertNotEqual(observer1, observer2)

    def test_not_equal_different_type(self):
        observer1 = SetItemObserver(notify=False, optional=False)
        imposter = mock.Mock()
        imposter.notify = False
        imposter.optional = False
        self.assertNotEqual(observer1, imposter)

    def test_equal_observers(self):
        observer1 = SetItemObserver(notify=False, optional=False)
        observer2 = SetItemObserver(notify=False, optional=False)
        self.assertEqual(observer1, observer2)
        self.assertEqual(hash(observer1), hash(observer2))

    def test_slots(self):
        observer = SetItemObserver(notify=True, optional=False)
        with self.assertRaises(AttributeError):
            observer.__dict__
        with self.assertRaises(AttributeError):
            observer.__weakref__

    def test_eval_repr_roundtrip(self):
        observer = SetItemObserver(notify=True, optional=False)
        self.assertEqual(eval(repr(observer)), observer)