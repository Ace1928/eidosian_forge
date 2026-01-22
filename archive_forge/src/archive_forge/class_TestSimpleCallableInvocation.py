from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
class TestSimpleCallableInvocation(LoggingComparisonTestCase):

    def test_callable_fn(self):

        def callback(x):
            return x
        self.assertEqual(Callable(callback)(3), 3)

    def test_callable_lambda(self):
        self.assertEqual(Callable(lambda x, y: x + y)(3, 5), 8)

    def test_callable_lambda_extras(self):
        substr = 'Ignoring extra positional argument'
        self.assertEqual(Callable(lambda x, y: x + y)(3, 5, 10), 8)
        self.log_handler.assertContains('WARNING', substr)

    def test_callable_lambda_extras_kwargs(self):
        substr = "['x'] overridden by keywords"
        self.assertEqual(Callable(lambda x, y: x + y)(3, 5, x=10), 15)
        self.log_handler.assertEndsWith('WARNING', substr)

    def test_callable_partial(self):
        self.assertEqual(Callable(partial(lambda x, y: x + y, x=4))(5), 9)

    def test_callable_class(self):
        self.assertEqual(Callable(CallableClass())(1, 2, 3, 4), 10)

    def test_callable_instance_method(self):
        assert Callable(CallableClass().someinstancemethod)(1, 2) == 3

    def test_callable_partial_instance_method(self):
        assert Callable(partial(CallableClass().someinstancemethod, x=1))(2) == 3

    def test_callable_paramfunc(self):
        self.assertEqual(Callable(ParamFunc)(3, b=5), 15)

    def test_callable_paramfunc_instance(self):
        self.assertEqual(Callable(ParamFunc.instance())(3, b=5), 15)