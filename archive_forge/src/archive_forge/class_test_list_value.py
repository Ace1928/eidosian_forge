import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class test_list_value(test_base2):

    def setUp(self):
        with self.assertWarns(DeprecationWarning):

            class list_value(HasTraits):
                list1 = Trait([2], TraitList(Trait([1, 2, 3, 4]), maxlen=4))
                list2 = Trait([2], TraitList(Trait([1, 2, 3, 4]), minlen=1, maxlen=4))
                alist = List()
        self.obj = list_value()
        self.last_event = None

    def tearDown(self):
        del self.last_event

    def del_range(self, list, index1, index2):
        del list[index1:index2]

    def del_extended_slice(self, list, index1, index2, step):
        del list[index1:index2:step]

    def check_list(self, list):
        self.assertEqual(list, [2])
        self.assertEqual(len(list), 1)
        list.append(3)
        self.assertEqual(len(list), 2)
        list[1] = 2
        self.assertEqual(list[1], 2)
        self.assertEqual(len(list), 2)
        list[0] = 1
        self.assertEqual(list[0], 1)
        self.assertEqual(len(list), 2)
        self.assertRaises(TraitError, self.indexed_assign, list, 0, 5)
        self.assertRaises(TraitError, list.append, 5)
        self.assertRaises(TraitError, list.extend, [1, 2, 3])
        list.extend([3, 4])
        self.assertEqual(list, [1, 2, 3, 4])
        self.assertRaises(TraitError, list.append, 1)
        self.assertRaises(ValueError, self.extended_slice_assign, list, 0, 4, 2, [4, 5, 6])
        del list[1]
        self.assertEqual(list, [1, 3, 4])
        del list[0]
        self.assertEqual(list, [3, 4])
        list[:0] = [1, 2]
        self.assertEqual(list, [1, 2, 3, 4])
        self.assertRaises(TraitError, self.indexed_range_assign, list, 0, 0, [1])
        del list[0:3]
        self.assertEqual(list, [4])
        self.assertRaises(TraitError, self.indexed_range_assign, list, 0, 0, [4, 5])

    def test_list1(self):
        self.check_list(self.obj.list1)

    def test_list2(self):
        self.check_list(self.obj.list2)
        self.assertRaises(TraitError, self.del_range, self.obj.list2, 0, 1)
        self.assertRaises(TraitError, self.del_extended_slice, self.obj.list2, 4, -5, -1)

    def assertLastTraitListEventEqual(self, index, removed, added):
        self.assertEqual(self.last_event.index, index)
        self.assertEqual(self.last_event.removed, removed)
        self.assertEqual(self.last_event.added, added)

    def test_trait_list_event(self):
        """ Record TraitListEvent behavior.
        """
        self.obj.alist = [1, 2, 3, 4]
        self.obj.on_trait_change(self._record_trait_list_event, 'alist_items')
        del self.obj.alist[0]
        self.assertLastTraitListEventEqual(0, [1], [])
        self.obj.alist.append(5)
        self.assertLastTraitListEventEqual(3, [], [5])
        self.obj.alist[0:2] = [6, 7]
        self.assertLastTraitListEventEqual(0, [2, 3], [6, 7])
        self.obj.alist[:2] = [4, 5]
        self.assertLastTraitListEventEqual(0, [6, 7], [4, 5])
        self.obj.alist[0:2:1] = [8, 9]
        self.assertLastTraitListEventEqual(0, [4, 5], [8, 9])
        self.obj.alist[0:2:1] = [8, 9]
        self.assertLastTraitListEventEqual(0, [8, 9], [8, 9])
        old_event = self.last_event
        self.obj.alist[4:] = []
        self.assertIs(self.last_event, old_event)
        self.obj.alist[0:4:2] = [10, 11]
        self.assertLastTraitListEventEqual(slice(0, 3, 2), [8, 4], [10, 11])
        del self.obj.alist[1:4:2]
        self.assertLastTraitListEventEqual(slice(1, 4, 2), [9, 5], [])
        self.obj.alist = [1, 2, 3, 4]
        del self.obj.alist[2:4]
        self.assertLastTraitListEventEqual(2, [3, 4], [])
        self.obj.alist[:0] = [5, 6, 7, 8]
        self.assertLastTraitListEventEqual(0, [], [5, 6, 7, 8])
        del self.obj.alist[:2]
        self.assertLastTraitListEventEqual(0, [5, 6], [])
        del self.obj.alist[0:2]
        self.assertLastTraitListEventEqual(0, [7, 8], [])
        del self.obj.alist[:]
        self.assertLastTraitListEventEqual(0, [1, 2], [])

    def _record_trait_list_event(self, object, name, old, new):
        self.last_event = new