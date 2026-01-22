import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
class TestTraitListObject(unittest.TestCase):

    def test_list_of_lists_pickle_with_notifier(self):

        class Foo:
            pass
        tl = TraitListObject(trait=List(), object=Foo(), name='foo', value=())
        self.assertEqual([tl.notifier], tl.notifiers)
        serialized = pickle.dumps(tl)
        tl_deserialized = pickle.loads(serialized)
        self.assertEqual([tl_deserialized.notifier], tl_deserialized.notifiers)

    def test_init_too_small(self):
        with self.assertRaises(TraitError):
            HasLengthConstrainedLists(at_least_two=[1])

    def test_init_too_large(self):
        with self.assertRaises(TraitError):
            HasLengthConstrainedLists(at_most_five=[1, 2, 3, 4, 5, 6])

    def test_init_from_iterable(self):

        class Foo:
            pass
        tl = TraitListObject(trait=List(), object=Foo(), name='foo', value=squares(5))
        self.assertEqual(tl, list(squares(5)))

    def test_delitem(self):
        foo = HasLengthConstrainedLists(at_most_five=[1, 23])
        del foo.at_most_five[1]
        self.assertEqual(foo.at_most_five, [1])

    def test_delitem_single_too_small(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2])
        with self.assertRaises(TraitError):
            del foo.at_least_two[0]
        self.assertEqual(foo.at_least_two, [1, 2])

    def test_delitem_slice_too_small(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2])
        with self.assertRaises(TraitError):
            del foo.at_least_two[:]
        self.assertEqual(foo.at_least_two, [1, 2])

    def test_delitem_from_empty(self):
        foo = HasLengthConstrainedLists()
        with self.assertRaises(IndexError):
            del foo.unconstrained[0]

    def test_iadd(self):
        foo = HasLengthConstrainedLists(at_most_five=[1, 2])
        foo.at_most_five += [6, 7, 8]
        self.assertEqual(foo.at_most_five, [1, 2, 6, 7, 8])

    def test_iadd_too_large(self):
        foo = HasLengthConstrainedLists(at_most_five=[1, 2, 3, 4])
        with self.assertRaises(TraitError):
            foo.at_most_five += [6, 7, 8]
        self.assertEqual(foo.at_most_five, [1, 2, 3, 4])

    def test_iadd_from_iterable(self):
        foo = HasLengthConstrainedLists(at_most_five=[1, 2])
        foo.at_most_five += squares(3)
        self.assertEqual(foo.at_most_five, [1, 2, 0, 1, 4])

    def test_imul(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2, 3])
        foo.at_least_two *= 2
        self.assertEqual(foo.at_least_two, [1, 2, 3, 1, 2, 3])

    def test_imul_too_small(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2, 3, 4])
        with self.assertRaises(TraitError):
            foo.at_least_two *= 0
        self.assertEqual(foo.at_least_two, [1, 2, 3, 4])

    def test_imul_too_large(self):
        foo = HasLengthConstrainedLists(at_most_five=[1, 2, 3, 4])
        with self.assertRaises(TraitError):
            foo.at_most_five *= 2
        self.assertEqual(foo.at_most_five, [1, 2, 3, 4])

    def test_imul_negative_multiplier(self):
        foo = HasLengthConstrainedLists(at_most_five=[1, 2, 3, 4])
        foo.at_most_five *= -10
        self.assertEqual(foo.at_most_five, [])

    def test_setitem_index(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2, 3, 4])
        foo.at_least_two[1] = 7
        self.assertEqual(foo.at_least_two, [1, 7, 3, 4])

    def test_setitem_slice(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2, 3, 4])
        foo.at_least_two[1:] = [6, 7]
        self.assertEqual(foo.at_least_two, [1, 6, 7])

    def test_setitem_extended_slice(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2, 3, 4])
        foo.at_least_two[1::2] = [6, 7]
        self.assertEqual(foo.at_least_two, [1, 6, 3, 7])

    def test_setitem_too_small(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2, 3, 4])
        with self.assertRaises(TraitError):
            foo.at_least_two[1:] = []
        self.assertEqual(foo.at_least_two, [1, 2, 3, 4])

    def test_setitem_too_large(self):
        foo = HasLengthConstrainedLists(at_most_five=[1, 2, 3, 4])
        with self.assertRaises(TraitError):
            foo.at_most_five[2:] = [10, 11, 12, 13]
        self.assertEqual(foo.at_most_five, [1, 2, 3, 4])

    def test_setitem_from_iterable(self):
        foo = HasLengthConstrainedLists(at_most_five=[1, 2])
        foo.at_most_five[:1] = squares(4)
        self.assertEqual(foo.at_most_five, [0, 1, 4, 9, 2])

    def test_setitem_extended_slice_bad_length(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2, 3, 4])
        with self.assertRaises(ValueError):
            foo.at_least_two[1::2] = squares(3)
        self.assertEqual(foo.at_least_two, [1, 2, 3, 4])

    def test_setitem_item_validation_failure(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2, 3, 4])
        with self.assertRaises(TraitError):
            foo.at_least_two[2:] = [5.0, 6.0]
        self.assertEqual(foo.at_least_two, [1, 2, 3, 4])

    def test_setitem_stop_lt_start(self):
        events = []
        foo = HasLengthConstrainedLists(at_least_two=[1, 2, 3, 4])
        foo.on_trait_change(lambda event: events.append(event), 'at_least_two_items')
        foo.at_least_two[4:2] = [5, 6, 7]
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.index, 4)
        self.assertEqual(event.removed, [])
        self.assertEqual(event.added, [5, 6, 7])

    def test_append(self):
        foo = HasLengthConstrainedLists(at_most_five=[1, 2, 3])
        foo.at_most_five.append(6)
        self.assertEqual(foo.at_most_five, [1, 2, 3, 6])

    def test_append_too_large(self):
        foo = HasLengthConstrainedLists(at_most_five=[1, 2, 3, 4, 5])
        with self.assertRaises(TraitError):
            foo.at_most_five.append(6)
        self.assertEqual(foo.at_most_five, [1, 2, 3, 4, 5])

    def test_clear(self):
        foo = HasLengthConstrainedLists(at_most_five=[1, 2, 3, 4])
        foo.at_most_five.clear()
        self.assertEqual(foo.at_most_five, [])

    def test_clear_too_small(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2, 3, 4])
        with self.assertRaises(TraitError):
            foo.at_least_two.clear()
        self.assertEqual(foo.at_least_two, [1, 2, 3, 4])

    def test_extend(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2, 3, 4])
        foo.at_least_two.extend([10, 11])
        self.assertEqual(foo.at_least_two, [1, 2, 3, 4, 10, 11])

    def test_extend_too_large(self):
        foo = HasLengthConstrainedLists(at_most_five=[1, 2, 3, 4])
        with self.assertRaises(TraitError):
            foo.at_most_five.extend([10, 11, 12])
        self.assertEqual(foo.at_most_five, [1, 2, 3, 4])

    def test_extend_from_iterable(self):
        foo = HasLengthConstrainedLists(at_most_five=[1, 2])
        foo.at_most_five.extend(squares(3))
        self.assertEqual(foo.at_most_five, [1, 2, 0, 1, 4])

    def test_insert(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2, 3, 4])
        foo.at_least_two.insert(3, 16)
        self.assertEqual(foo.at_least_two, [1, 2, 3, 16, 4])

    def test_insert_too_large(self):
        foo = HasLengthConstrainedLists(at_most_five=[1, 2, 3, 4, 5])
        with self.assertRaises(TraitError):
            foo.at_most_five.insert(3, 16)
        with self.assertRaises(TraitError):
            foo.at_most_five.insert(-10, 16)
        with self.assertRaises(TraitError):
            foo.at_most_five.insert(10, 16)
        self.assertEqual(foo.at_most_five, [1, 2, 3, 4, 5])

    def test_pop(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2, 6])
        foo.at_least_two.pop()
        self.assertEqual(foo.at_least_two, [1, 2])

    def test_pop_too_small(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2])
        with self.assertRaises(TraitError):
            foo.at_least_two.pop()
        with self.assertRaises(TraitError):
            foo.at_least_two.pop(0)
        with self.assertRaises(TraitError):
            foo.at_least_two.pop(10)
        self.assertEqual(foo.at_least_two, [1, 2])

    def test_pop_from_empty(self):
        foo = HasLengthConstrainedLists()
        with self.assertRaises(IndexError):
            foo.unconstrained.pop()
        with self.assertRaises(IndexError):
            foo.unconstrained.pop(10)

    def test_remove(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2, 6, 4])
        foo.at_least_two.remove(2)
        self.assertEqual(foo.at_least_two, [1, 6, 4])

    def test_remove_too_small(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2])
        with self.assertRaises(TraitError):
            foo.at_least_two.remove(1)
        with self.assertRaises(TraitError):
            foo.at_least_two.remove(2.0)
        with self.assertRaises(TraitError):
            foo.at_least_two.remove(10)
        self.assertEqual(foo.at_least_two, [1, 2])

    def test_remove_from_empty(self):
        foo = HasLengthConstrainedLists()
        with self.assertRaises(ValueError):
            foo.unconstrained.remove(35)

    def test_length_violation_error_message(self):
        foo = HasLengthConstrainedLists(at_least_two=[1, 2])
        with self.assertRaises(TraitError) as exc_cm:
            foo.at_least_two.remove(1)
        exc_message = str(exc_cm.exception)
        self.assertIn("'at_least_two' trait", exc_message)
        self.assertIn('HasLengthConstrainedLists instance', exc_message)
        self.assertIn('an integer', exc_message)
        self.assertIn('at least 2 items', exc_message)

    def test_dead_object_reference(self):
        foo = HasLengthConstrainedLists(at_most_five=[1, 2, 3, 4])
        list_object = foo.at_most_five
        del foo
        list_object.append(5)
        self.assertEqual(list_object, [1, 2, 3, 4, 5])
        with self.assertRaises(TraitError):
            list_object.append(4)