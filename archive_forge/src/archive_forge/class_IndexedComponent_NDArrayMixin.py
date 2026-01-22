import inspect
import logging
import sys
import textwrap
import pyomo.core.expr as EXPR
import pyomo.core.base as BASE
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.component import Component, ActiveComponent
from pyomo.core.base.config import PyomoOptions
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_set
from pyomo.core.expr.numeric_expr import _ndarray
from pyomo.core.pyomoobject import PyomoObject
from pyomo.common import DeveloperError
from pyomo.common.autoslots import fast_deepcopy
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import native_types
from pyomo.common.sorting import sorted_robust
from collections.abc import Sequence
class IndexedComponent_NDArrayMixin(object):
    """Support using IndexedComponent with numpy.ndarray

    This IndexedComponent mixin class adds support for implicitly using
    the IndexedComponent as a term in an expression with numpy ndarray
    objects.

    """

    def __array__(self, dtype=None):
        if not self.is_indexed():
            ans = _ndarray.NumericNDArray(shape=(1,), dtype=object)
            ans[0] = self
            return ans
        _dim = self.dim()
        if _dim is None:
            raise TypeError('Cannot convert a non-dimensioned Pyomo IndexedComponent (%s) into a numpy array' % (self,))
        bounds = self.index_set().bounds()
        if not isinstance(bounds[0], Sequence):
            bounds = ((bounds[0],), (bounds[1],))
        if any((b != 0 for b in bounds[0])):
            raise TypeError('Cannot convert a Pyomo IndexedComponent (%s) with bounds [%s, %s] into a numpy array' % (self, bounds[0], bounds[1]))
        shape = tuple((b + 1 for b in bounds[1]))
        ans = _ndarray.NumericNDArray(shape=shape, dtype=object)
        for k, v in self.items():
            ans[k] = v
        return ans

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _ndarray.NumericNDArray.__array_ufunc__(None, ufunc, method, *inputs, **kwargs)