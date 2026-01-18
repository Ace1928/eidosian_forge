from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_associativity(self):
    o1 = self.el1 * self.el2 * self.el3
    o2 = self.el1 * self.el2 * self.el3
    o3 = self.el1 * (self.el2 * self.el3)
    self.assertEqual(o1.keys(), o2.keys())
    self.assertEqual(o2.keys(), o3.keys())