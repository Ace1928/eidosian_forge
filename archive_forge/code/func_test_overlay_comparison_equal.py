from holoviews import Element
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_comparison_equal(self):
    t1 = self.el1 * self.el2
    t2 = self.el1 * self.el2
    self.assertEqual(t1, t2)