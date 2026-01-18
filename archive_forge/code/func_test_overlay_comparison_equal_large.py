from holoviews import Element
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_comparison_equal_large(self):
    t1 = self.el1 * self.el2 * self.el3 * self.el4
    t2 = self.el1 * self.el2 * self.el3 * self.el4
    self.assertEqual(t1, t2)