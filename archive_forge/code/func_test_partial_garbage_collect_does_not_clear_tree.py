import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
def test_partial_garbage_collect_does_not_clear_tree(self):
    obj = HoloMap({0: ExampleElement([]), 1: ExampleElement([])}).opts(style_opt1='A')
    obj.pop(0)
    gc.collect()
    custom_options = Store._custom_options['backend_1']
    self.assertIn(obj.last.id, custom_options)
    self.assertEqual(len(custom_options), 1)
    obj.pop(1)
    gc.collect()
    self.assertEqual(len(custom_options), 0)