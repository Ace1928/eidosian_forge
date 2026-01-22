import logging
from pyomo.common import DeveloperError
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import numpy as np
from pyomo.common.dependencies.scipy import spatial
from pyomo.contrib.piecewise.piecewise_linear_expression import (
from pyomo.core import Any, NonNegativeIntegers, value, Var
from pyomo.core.base.block import _BlockData, Block
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.expression import Expression
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import UnindexedComponent_set
from pyomo.core.base.initializer import Initializer
import pyomo.core.expr as EXPR
@ModelComponentFactory.register('Multidimensional piecewise linear function')
class PiecewiseLinearFunction(Block):
    """A piecewise linear function, which may be defined over an index.

    Can be specified in one of several ways:
        1) List of points and a nonlinear function to approximate. In
           this case, the points will be used to derive a triangulation
           of the part of the domain of interest, and a linear function
           approximating the given function will be calculated for each
           of the simplices in the triangulation. In this case, scipy is
           required (for multivariate functions).
        2) List of simplices and a nonlinear function to approximate. In
           this case, a linear function approximating the given function
           will be calculated for each simplex. For multivariate functions,
           numpy is required.
        3) List of simplices and list of functions that return linear function
           expressions. These are the desired piecewise functions
           corresponding to each simplex.
        4) Mapping of function values to points of the domain, allowing for
           the construction of a piecewise linear function from tabular data.

    Args:
        function: Nonlinear function to approximate: must be callable
        function_rule: Function that returns a nonlinear function to
            approximate for each index in an IndexedPiecewiseLinearFunction
        points: List of points in the same dimension as the domain of the
            function being approximated. Note that if the pieces of the
            function are specified this way, we require scipy.
        simplices: A list of lists of points, where each list specifies the
            extreme points of a a simplex over which the nonlinear function
            will be approximated as a linear function.
        linear_functions: A list of functions, each of which returns an
            expression for a linear function of the arguments.
        tabular_data: A dictionary mapping values of the nonlinear function
            to points in the domain
    """
    _ComponentDataClass = PiecewiseLinearFunctionData
    _handlers = {}

    def __new__(cls, *args, **kwds):
        if cls is not PiecewiseLinearFunction:
            return super(PiecewiseLinearFunction, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return PiecewiseLinearFunction.__new__(ScalarPiecewiseLinearFunction)
        else:
            return IndexedPiecewiseLinearFunction.__new__(IndexedPiecewiseLinearFunction)

    def __init__(self, *args, **kwargs):
        _func_arg = kwargs.pop('function', None)
        _func_rule_arg = kwargs.pop('function_rule', None)
        _points_arg = kwargs.pop('points', None)
        _simplices_arg = kwargs.pop('simplices', None)
        _linear_functions = kwargs.pop('linear_functions', None)
        _tabular_data_arg = kwargs.pop('tabular_data', None)
        _tabular_data_rule_arg = kwargs.pop('tabular_data_rule', None)
        kwargs.setdefault('ctype', PiecewiseLinearFunction)
        Block.__init__(self, *args, **kwargs)
        self._func = _func_arg
        self._func_rule = Initializer(_func_rule_arg)
        self._points_rule = Initializer(_points_arg, treat_sequences_as_mappings=False)
        self._simplices_rule = Initializer(_simplices_arg, treat_sequences_as_mappings=False)
        self._linear_funcs_rule = Initializer(_linear_functions, treat_sequences_as_mappings=False)
        self._tabular_data = _tabular_data_arg
        self._tabular_data_rule = Initializer(_tabular_data_rule_arg, treat_sequences_as_mappings=False)

    def _get_dimension_from_points(self, points):
        if len(points) < 1:
            raise ValueError('Cannot construct PiecewiseLinearFunction from points list of length 0.')
        if hasattr(points[0], '__len__'):
            dimension = len(points[0])
        else:
            dimension = 1
        return dimension

    def _construct_simplices_from_multivariate_points(self, obj, points, dimension):
        try:
            triangulation = spatial.Delaunay(points)
        except (spatial.QhullError, ValueError) as error:
            logger.error('Unable to triangulate the set of input points.')
            raise
        obj._points = [pt for pt in map(tuple, triangulation.points)]
        obj._simplices = []
        for simplex in triangulation.simplices:
            points = triangulation.points[simplex].transpose()
            if np.linalg.matrix_rank(points[:, 1:] - np.append(points[:, :dimension - 1], points[:, [0]], axis=1)) == dimension:
                obj._simplices.append(tuple(sorted(simplex)))
        for pt in triangulation.coplanar:
            logger.info('The Delaunay triangulation dropped the point with index %s from the triangulation.' % pt[0])

    def _construct_one_dimensional_simplices_from_points(self, obj, points):
        points.sort()
        obj._simplices = []
        for i in range(len(points) - 1):
            obj._simplices.append((i, i + 1))
            obj._points.append((points[i],))
        obj._points.append((points[-1],))

    @_define_handler(_handlers, True, True, False, False, False)
    def _construct_from_function_and_points(self, obj, parent, nonlinear_function):
        idx = obj._index
        points = self._points_rule(parent, idx)
        dimension = self._get_dimension_from_points(points)
        if dimension == 1:
            self._construct_one_dimensional_simplices_from_points(obj, points)
            return self._construct_from_univariate_function_and_segments(obj, nonlinear_function)
        self._construct_simplices_from_multivariate_points(obj, points, dimension)
        return self._construct_from_function_and_simplices(obj, parent, nonlinear_function, simplices_are_user_defined=False)

    def _construct_from_univariate_function_and_segments(self, obj, func):
        for idx1, idx2 in obj._simplices:
            x1 = obj._points[idx1][0]
            x2 = obj._points[idx2][0]
            y = {x: func(x) for x in [x1, x2]}
            slope = (y[x2] - y[x1]) / (x2 - x1)
            intercept = y[x1] - slope * x1
            obj._linear_functions.append(_univariate_linear_functor(slope, intercept))
        return obj

    @_define_handler(_handlers, True, False, True, False, False)
    def _construct_from_function_and_simplices(self, obj, parent, nonlinear_function, simplices_are_user_defined=True):
        if obj._simplices is None:
            obj._get_simplices_from_arg(self._simplices_rule(parent, obj._index))
        simplices = obj._simplices
        if len(simplices) < 1:
            raise ValueError('Cannot construct PiecewiseLinearFunction with empty list of simplices')
        dimension = len(simplices[0]) - 1
        if dimension == 1:
            return self._construct_from_univariate_function_and_segments(obj, nonlinear_function)
        A = np.ones((dimension + 2, dimension + 2))
        b = np.zeros(dimension + 2)
        b[-1] = 1
        for num_piece, simplex in enumerate(simplices):
            for i, pt_idx in enumerate(simplex):
                pt = obj._points[pt_idx]
                for j, val in enumerate(pt):
                    A[i, j] = val
                A[i, j + 1] = nonlinear_function(*pt)
            A[i + 1, :] = 0
            A[i + 1, dimension] = -1
            try:
                normal = np.linalg.solve(A, b)
            except np.linalg.LinAlgError as e:
                logger.warning('LinAlgError: %s' % e)
                msg = 'When calculating the hyperplane approximation over the simplex with index %s, the matrix was unexpectedly singular. This likely means that this simplex is degenerate' % num_piece
                if simplices_are_user_defined:
                    raise ValueError(msg)
                raise DeveloperError(msg + ' and that it should have been filtered out of the triangulation')
            obj._linear_functions.append(_multivariate_linear_functor(normal))
        return obj

    @_define_handler(_handlers, False, False, True, True, False)
    def _construct_from_linear_functions_and_simplices(self, obj, parent, nonlinear_function):
        obj._get_simplices_from_arg(self._simplices_rule(parent, obj._index))
        obj._linear_functions = [f for f in self._linear_funcs_rule(parent, obj._index)]
        return obj

    @_define_handler(_handlers, False, False, False, False, True)
    def _construct_from_tabular_data(self, obj, parent, nonlinear_function):
        idx = obj._index
        tabular_data = self._tabular_data
        if tabular_data is None:
            tabular_data = self._tabular_data_rule(parent, idx)
        points = [pt for pt in tabular_data.keys()]
        dimension = self._get_dimension_from_points(points)
        if dimension == 1:
            self._construct_one_dimensional_simplices_from_points(obj, points)
            return self._construct_from_univariate_function_and_segments(obj, _tabular_data_functor(tabular_data, tupleize=True))
        self._construct_simplices_from_multivariate_points(obj, points, dimension)
        return self._construct_from_function_and_simplices(obj, parent, _tabular_data_functor(tabular_data))

    def _getitem_when_not_present(self, index):
        if index is None and (not self.is_indexed()):
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(component=self)
        obj._index = index
        parent = obj.parent_block()
        nonlinear_function = None
        if self._func_rule is not None:
            nonlinear_function = self._func_rule(parent, index)
        elif self._func is not None:
            nonlinear_function = self._func
        handler = self._handlers.get((nonlinear_function is not None, self._points_rule is not None, self._simplices_rule is not None, self._linear_funcs_rule is not None, self._tabular_data is not None or self._tabular_data_rule is not None))
        if handler is None:
            raise ValueError('Unsupported set of arguments given for constructing PiecewiseLinearFunction. Expected a nonlinear function and a listof breakpoints, a nonlinear function and a list of simplices, a list of linear functions and a list of corresponding simplices, or a dictionary mapping points to nonlinear function values.')
        return handler(self, obj, parent, nonlinear_function)