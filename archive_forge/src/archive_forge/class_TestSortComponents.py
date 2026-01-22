import pyomo.common.unittest as unittest
from pyomo.core.base.enums import SortComponents
class TestSortComponents(unittest.TestCase):

    def test_mappings(self):
        self.assertEqual(SortComponents(True), SortComponents.SORTED_INDICES | SortComponents.ALPHABETICAL)
        self.assertEqual(SortComponents(False), SortComponents.UNSORTED)
        self.assertEqual(SortComponents(None), SortComponents.UNSORTED)
        with self.assertRaisesRegex(ValueError, '(999 is not a valid SortComponents)|(invalid value 999)'):
            SortComponents(999)
        self.assertEqual(SortComponents._missing_(False), SortComponents.UNSORTED)

    def test_sorter(self):
        self.assertEqual(SortComponents.sorter(), SortComponents.UNSORTED)
        self.assertEqual(SortComponents.sorter(True, False), SortComponents.ALPHABETICAL)
        self.assertEqual(SortComponents.sorter(False, True), SortComponents.SORTED_INDICES)
        self.assertEqual(SortComponents.sorter(True, True), SortComponents.ALPHABETICAL | SortComponents.SORTED_INDICES)

    def test_methods(self):
        self.assertEqual(SortComponents.default(), SortComponents.UNSORTED)
        self.assertTrue(SortComponents.sort_names(SortComponents.sorter(True, True)))
        self.assertTrue(SortComponents.sort_names(SortComponents.sorter(True, False)))
        self.assertFalse(SortComponents.sort_names(SortComponents.sorter(False, True)))
        self.assertFalse(SortComponents.sort_names(SortComponents.sorter(False, False)))
        self.assertTrue(SortComponents.sort_indices(SortComponents.sorter(True, True)))
        self.assertFalse(SortComponents.sort_indices(SortComponents.sorter(True, False)))
        self.assertTrue(SortComponents.sort_indices(SortComponents.sorter(False, True)))
        self.assertFalse(SortComponents.sort_indices(SortComponents.sorter(False, False)))