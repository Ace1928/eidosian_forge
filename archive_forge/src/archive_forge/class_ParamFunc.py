from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
class ParamFunc(param.ParameterizedFunction):
    a = param.Integer(default=1)
    b = param.Number(default=1)

    def __call__(self, a, **params):
        p = param.ParamOverrides(self, params)
        return a * p.b