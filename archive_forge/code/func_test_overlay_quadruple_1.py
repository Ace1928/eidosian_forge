from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_quadruple_1(self):
    t = self.el1 * self.el1 * self.el1 * self.el1
    self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'II'), ('Element', 'III'), ('Element', 'IV')])