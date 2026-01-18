from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_triple_overlay_values(self):
    t = self.el1 * self.el2 * self.el3
    self.assertEqual(t.values(), [self.el1, self.el2, self.el3])