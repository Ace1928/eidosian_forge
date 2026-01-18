from holoviews import Curve, Element, Layout, Overlay, Store
from holoviews.core.pprint import PrettyPrinter
from holoviews.element.comparison import ComparisonTestCase
from .test_dimensioned import CustomBackendTestCase, ExampleElement
def test_overlay_repr1(self):
    expected = ':Overlay\n   .Value.Label :Element\n   .Value.I     :Element'
    o = self.element1 * self.element2
    r = PrettyPrinter.pprint(o)
    self.assertEqual(r, expected)