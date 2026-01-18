from holoviews import Curve, Element, Layout, Overlay, Store
from holoviews.core.pprint import PrettyPrinter
from holoviews.element.comparison import ComparisonTestCase
from .test_dimensioned import CustomBackendTestCase, ExampleElement
def test_overlay_nested_options(self):
    overlay = (ExampleElement(None) * ExampleElement(None)).opts('ExampleElement', plot_opt1='A', style_opt1='A')
    r = self.pprinter.pprint(overlay)
    self.assertEqual(r, ":Overlay\n   .Element.I  :ExampleElement\n    | Options(plot_opt1='A', style_opt1='A')\n   .Element.II :ExampleElement\n    | Options(plot_opt1='A', style_opt1='A')")