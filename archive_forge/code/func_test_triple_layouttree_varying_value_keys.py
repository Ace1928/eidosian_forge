from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_triple_layouttree_varying_value_keys(self):
    t = self.el1 + self.el4 + self.el2 + self.el3
    expected_keys = [('Element', 'I'), ('ValA', 'I'), ('Element', 'II'), ('Element', 'III')]
    self.assertEqual(t.keys(), expected_keys)