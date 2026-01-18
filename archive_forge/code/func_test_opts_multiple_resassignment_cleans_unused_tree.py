import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
def test_opts_multiple_resassignment_cleans_unused_tree(self):
    obj = HoloMap({0: ExampleElement([]), 1: ExampleElement([])}).opts(style_opt1='A').opts(plot_opt1='B')
    custom_options = Store._custom_options['backend_1']
    self.assertIn(obj.last.id, custom_options)
    self.assertEqual(len(custom_options), 2)
    for o in obj:
        o.id = None
    self.assertEqual(len(custom_options), 0)