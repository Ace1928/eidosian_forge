from holoviews import Curve, Element, Layout, Overlay, Store
from holoviews.core.pprint import PrettyPrinter
from holoviews.element.comparison import ComparisonTestCase
from .test_dimensioned import CustomBackendTestCase, ExampleElement
class PrettyPrintTest(ComparisonTestCase):

    def setUp(self):
        self.element1 = Element(None, group='Value', label='Label')
        self.element2 = Element(None, group='Value', label='')

    def test_element_repr1(self):
        r = PrettyPrinter.pprint(self.element1)
        self.assertEqual(r, ':Element')

    def test_overlay_repr1(self):
        expected = ':Overlay\n   .Value.Label :Element\n   .Value.I     :Element'
        o = self.element1 * self.element2
        r = PrettyPrinter.pprint(o)
        self.assertEqual(r, expected)

    def test_curve_pprint_repr(self):
        expected = "':Curve   [x]   (y)'"
        r = PrettyPrinter.pprint(Curve([1, 2, 3]))
        self.assertEqual(repr(r), expected)