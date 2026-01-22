from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
class DimensionedComparisonTestCase(ComparisonTestCase):

    def setUp(self):
        super().setUp()
        self.value_list1 = [Dimension('val1')]
        self.value_list2 = [Dimension('val2')]
        self.key_list1 = [Dimension('key1')]
        self.key_list2 = [Dimension('key2')]
        self.dimensioned1 = Dimensioned('data1', vdims=self.value_list1, kdims=self.key_list1)
        self.dimensioned2 = Dimensioned('data2', vdims=self.value_list2, kdims=self.key_list1)
        self.dimensioned3 = Dimensioned('data3', vdims=self.value_list1, kdims=self.key_list2)
        self.dimensioned4 = Dimensioned('data4', vdims=[], kdims=self.key_list1)
        self.dimensioned5 = Dimensioned('data5', vdims=self.value_list1, kdims=[])
        self.dimensioned6 = Dimensioned('data6', group='foo', vdims=self.value_list1, kdims=self.key_list1)
        self.dimensioned7 = Dimensioned('data7', group='foo', label='bar', vdims=self.value_list1, kdims=self.key_list1)

    def test_dimensioned_comparison_equal(self):
        """Note that the data is not compared at the Dimensioned level"""
        self.assertEqual(self.dimensioned1, Dimensioned('other_data', vdims=self.value_list1, kdims=self.key_list1))

    def test_dimensioned_comparison_unequal_value_dims(self):
        try:
            self.assertEqual(self.dimensioned1, self.dimensioned2)
        except AssertionError as e:
            self.assertEqual(str(e), 'Dimension names mismatched: val1 != val2')

    def test_dimensioned_comparison_unequal_key_dims(self):
        try:
            self.assertEqual(self.dimensioned1, self.dimensioned3)
        except AssertionError as e:
            self.assertEqual(str(e), 'Dimension names mismatched: key1 != key2')

    def test_dimensioned_comparison_unequal_value_dim_lists(self):
        try:
            self.assertEqual(self.dimensioned1, self.dimensioned4)
        except AssertionError as e:
            self.assertEqual(str(e), 'Value dimension list mismatched')

    def test_dimensioned_comparison_unequal_key_dim_lists(self):
        try:
            self.assertEqual(self.dimensioned1, self.dimensioned5)
        except AssertionError as e:
            self.assertEqual(str(e), 'Key dimension list mismatched')

    def test_dimensioned_comparison_unequal_group(self):
        try:
            self.assertEqual(self.dimensioned1, self.dimensioned6)
        except AssertionError as e:
            self.assertEqual(str(e), 'Group labels mismatched.')

    def test_dimensioned_comparison_unequal_label(self):
        try:
            self.assertEqual(self.dimensioned6, self.dimensioned7)
        except AssertionError as e:
            self.assertEqual(str(e), 'Labels mismatched.')