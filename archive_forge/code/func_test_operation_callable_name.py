from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
def test_operation_callable_name(self):
    opcallable = OperationCallable(lambda x: x, operation=contours.instance())
    self.assertEqual(Callable(opcallable).name, 'contours')