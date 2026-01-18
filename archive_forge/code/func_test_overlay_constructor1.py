from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_constructor1(self):
    t = Overlay([self.el1])
    self.assertEqual(t.keys(), [('Element', 'I')])