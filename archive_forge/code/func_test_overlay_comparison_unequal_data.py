from holoviews import Element
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_comparison_unequal_data(self):
    t1 = self.el1 * self.el2
    t2 = self.el1 * self.el3
    try:
        self.assertEqual(t1, t2)
    except AssertionError as e:
        self.assertEqual(str(e), "'data2' != 'data3'")