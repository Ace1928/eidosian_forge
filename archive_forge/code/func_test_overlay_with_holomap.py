from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_with_holomap(self):
    overlay = Overlay([('Custom', self.el6)])
    composite = overlay * HoloMap({0: Element(None, group='HoloMap')})
    self.assertEqual(composite.last.keys(), [('Custom', 'LabelA'), ('HoloMap', 'I')])