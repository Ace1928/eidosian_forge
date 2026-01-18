from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_varying_value_keys(self):
    t = self.el1 * self.el4
    self.assertEqual(t.keys(), [('Element', 'I'), ('ValA', 'I')])