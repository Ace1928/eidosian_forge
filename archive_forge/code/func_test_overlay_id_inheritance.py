from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_id_inheritance(self):
    overlay = Overlay([], id=1)
    self.assertEqual(overlay.clone().id, 1)
    self.assertEqual(overlay.clone()._plot_id, overlay._plot_id)
    self.assertNotEqual(overlay.clone([])._plot_id, overlay._plot_id)