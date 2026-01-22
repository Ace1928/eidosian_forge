from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
class TestCallableArgspec(ComparisonTestCase):

    def test_callable_fn_argspec(self):

        def callback(x):
            return x
        self.assertEqual(Callable(callback).argspec.args, ['x'])
        self.assertEqual(Callable(callback).argspec.keywords, None)

    def test_callable_lambda_argspec(self):
        self.assertEqual(Callable(lambda x, y: x + y).argspec.args, ['x', 'y'])
        self.assertEqual(Callable(lambda x, y: x + y).argspec.keywords, None)

    def test_callable_partial_argspec(self):
        self.assertEqual(Callable(partial(lambda x, y: x + y, x=4)).argspec.args, ['y'])
        self.assertEqual(Callable(partial(lambda x, y: x + y, x=4)).argspec.keywords, None)

    def test_callable_class_argspec(self):
        self.assertEqual(Callable(CallableClass()).argspec.args, [])
        self.assertEqual(Callable(CallableClass()).argspec.keywords, None)
        self.assertEqual(Callable(CallableClass()).argspec.varargs, 'testargs')

    def test_callable_instance_method(self):
        assert Callable(CallableClass().someinstancemethod).argspec.args == ['x', 'y']
        assert Callable(CallableClass().someinstancemethod).argspec.keywords is None

    def test_callable_partial_instance_method(self):
        assert Callable(partial(CallableClass().someinstancemethod, x=1)).argspec.args == ['y']
        assert Callable(partial(CallableClass().someinstancemethod, x=1)).argspec.keywords is None

    def test_callable_paramfunc_argspec(self):
        self.assertEqual(Callable(ParamFunc).argspec.args, ['a'])
        self.assertEqual(Callable(ParamFunc).argspec.keywords, 'params')
        self.assertEqual(Callable(ParamFunc).argspec.varargs, None)

    def test_callable_paramfunc_instance_argspec(self):
        self.assertEqual(Callable(ParamFunc.instance()).argspec.args, ['a'])
        self.assertEqual(Callable(ParamFunc.instance()).argspec.keywords, 'params')
        self.assertEqual(Callable(ParamFunc.instance()).argspec.varargs, None)