import collections
import enum
import functools
import itertools
import logging
import operator
import sys
from pyomo.common.collections import Sequence, ComponentMap, ComponentSet
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.expression import _ExpressionData
from pyomo.core.expr.numvalue import is_fixed, value
import pyomo.core.expr as EXPR
import pyomo.core.kernel as kernel
class BeforeChildDispatcher(collections.defaultdict):
    """Dispatcher for handling the :py:class:`StreamBasedExpressionVisitor`
    `beforeChild` callback

    This dispatcher implements a specialization of :py:`defaultdict`
    that supports automatic type registration.  Any missing types will
    return the :py:meth:`register_dispatcher` method, which (when called
    as a callback) will interrogate the type, identify the appropriate
    callback, add the callback to the dict, and return the result of
    calling the callback.  As the callback is added to the dict, no type
    will incur the overhead of `register_dispatcher` more than once.

    Note that all dispatchers are implemented as `staticmethod`
    functions to avoid the (unnecessary) overhead of binding to the
    dispatcher object.

    """
    __slots__ = ()

    def __missing__(self, key):
        return self.register_dispatcher

    def register_dispatcher(self, visitor, child):
        child_type = type(child)
        if child_type in native_numeric_types:
            self[child_type] = self._before_native_numeric
        elif child_type in native_logical_types:
            self[child_type] = self._before_native_logical
        elif issubclass(child_type, str):
            self[child_type] = self._before_string
        elif child_type in native_types:
            if issubclass(child_type, tuple(native_complex_types)):
                self[child_type] = self._before_complex
            else:
                self[child_type] = self._before_invalid
        elif not hasattr(child, 'is_expression_type'):
            if check_if_numeric_type(child):
                self[child_type] = self._before_native_numeric
            else:
                self[child_type] = self._before_invalid
        elif not child.is_expression_type():
            if child.is_potentially_variable():
                self[child_type] = self._before_var
            else:
                self[child_type] = self._before_param
        elif not child.is_potentially_variable():
            self[child_type] = self._before_npv
            pv_base_type = child.potentially_variable_base_class()
            if pv_base_type not in self:
                try:
                    child.__class__ = pv_base_type
                    self.register_dispatcher(visitor, child)
                finally:
                    child.__class__ = child_type
        elif issubclass(child_type, _named_subexpression_types) or child_type is kernel.expression.noclone:
            self[child_type] = self._before_named_expression
        else:
            self[child_type] = self._before_general_expression
        return self[child_type](visitor, child)

    @staticmethod
    def _before_general_expression(visitor, child):
        return (True, None)

    @staticmethod
    def _before_native_numeric(visitor, child):
        return (False, (_CONSTANT, child))

    @staticmethod
    def _before_native_logical(visitor, child):
        return (False, (_CONSTANT, InvalidNumber(child, f'{child!r} ({type(child).__name__}) is not a valid numeric type')))

    @staticmethod
    def _before_complex(visitor, child):
        return (False, (_CONSTANT, complex_number_error(child, visitor, child)))

    @staticmethod
    def _before_invalid(visitor, child):
        return (False, (_CONSTANT, InvalidNumber(child, f'{child!r} ({type(child).__name__}) is not a valid numeric type')))

    @staticmethod
    def _before_string(visitor, child):
        return (False, (_CONSTANT, InvalidNumber(child, f'{child!r} ({type(child).__name__}) is not a valid numeric type')))

    @staticmethod
    def _before_npv(visitor, child):
        try:
            return (False, (_CONSTANT, visitor.check_constant(visitor.evaluate(child), child)))
        except (ValueError, ArithmeticError):
            return (True, None)

    @staticmethod
    def _before_param(visitor, child):
        return (False, (_CONSTANT, visitor.check_constant(child.value, child)))