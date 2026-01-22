import unittest
from traits.api import (
class TestTraitEvent(unittest.TestCase):

    def setUp(self):
        self.foo = Foo()

    def test_list_repr(self):
        self.foo.alist[::2] = [4, 5]
        event = self.foo.event
        event_str = 'TraitListEvent(index=slice(0, 3, 2), removed=[1, 3], added=[4, 5])'
        self.assertEqual(repr(event), event_str)
        self.assertIsInstance(eval(repr(event)), TraitListEvent)

    def test_list_event_kwargs_only(self):
        with self.assertRaises(TypeError):
            TraitListEvent(slice(0, 3, 2), [1, 3], [4, 5])

    def test_dict_event_kwargs_only(self):
        with self.assertRaises(TypeError):
            TraitDictEvent({}, {'black': 0}, {'blue': 2})

    def test_dict_event_repr(self):
        self.foo.adict.update({'blue': 10, 'black': 0})
        event = self.foo.event
        event_str = "TraitDictEvent(removed={}, added={'black': 0}, changed={'blue': 0})"
        self.assertEqual(repr(event), event_str)
        self.assertIsInstance(eval(repr(event)), TraitDictEvent)

    def test_set_event_kwargs_only(self):
        with self.assertRaises(TypeError):
            TraitSetEvent({3}, {4})

    def test_set_event_repr(self):
        self.foo.aset.symmetric_difference_update({3, 4})
        event = self.foo.event
        event_str = 'TraitSetEvent(removed={3}, added={4})'
        self.assertEqual(repr(event), event_str)
        self.assertIsInstance(eval(repr(event)), TraitSetEvent)