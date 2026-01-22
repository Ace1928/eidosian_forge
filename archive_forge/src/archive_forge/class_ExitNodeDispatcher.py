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
class ExitNodeDispatcher(collections.defaultdict):
    """Dispatcher for handling the :py:class:`StreamBasedExpressionVisitor`
    `exitNode` callback

    This dispatcher implements a specialization of :py:`defaultdict`
    that supports automatic type registration.  Any missing types will
    return the :py:meth:`register_dispatcher` method, which (when called
    as a callback) will interrogate the type, identify the appropriate
    callback, add the callback to the dict, and return the result of
    calling the callback.  As the callback is added to the dict, no type
    will incur the overhead of `register_dispatcher` more than once.

    Note that in this case, the client is expected to register all
    non-NPV expression types.  The auto-registration is designed to only
    handle two cases:
    - Auto-detection of user-defined Named Expression types
    - Automatic mappimg of NPV expressions to their equivalent non-NPV handlers

    """
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(None, *args, **kwargs)

    def __missing__(self, key):
        if type(key) is tuple:
            node_class = key[0]
        else:
            node_class = key
        bases = node_class.__mro__
        if issubclass(node_class, _named_subexpression_types) or node_class is kernel.expression.noclone:
            bases = [Expression]
        fcn = None
        for base_type in bases:
            if isinstance(key, tuple):
                base_key = (base_type,) + key[1:]
                cache = len(key) <= 4
            else:
                base_key = base_type
                cache = True
            if base_key in self:
                fcn = self[base_key]
            elif base_type in self:
                fcn = self[base_type]
            elif any(((k[0] if type(k) is tuple else k) is base_type for k in self)):
                raise DeveloperError(f"Base expression key '{base_key}' not found when inserting dispatcher for node '{node_class.__name__}' while walking expression tree.")
        if fcn is None:
            fcn = self.unexpected_expression_type
        if cache:
            self[key] = fcn
        return fcn

    def unexpected_expression_type(self, visitor, node, *arg):
        raise DeveloperError(f"Unexpected expression node type '{type(node).__name__}' found while walking expression tree in {type(visitor).__name__}.")