import inspect
import logging
import sys
from copy import deepcopy
from collections import deque
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import DeveloperError, TemplateExpressionError
from pyomo.common.numeric_types import (
import pyomo.core.expr.expr_common as common
from pyomo.core.expr.symbol_map import SymbolMap
class ExpressionValueVisitor(object):
    """
    Note:
        This class is a customization of the PyUtilib :class:`ValueVisitor
        <pyutilib.misc.visitor.ValueVisitor>` class that is tailored
        to efficiently walk Pyomo expression trees.  However, this class
        is not a subclass of the PyUtilib :class:`ValueVisitor
        <pyutilib.misc.visitor.ValueVisitor>` class because all key methods
        are reimplemented.
    """

    def visit(self, node, values):
        """
        Visit a node in a tree and compute its value using
        the values of its children.

        This method should be over-written by a user
        that is creating a sub-class.

        Args:
            node: a node in a tree
            values: a list of values of this node's children

        Returns:
            The *value* for this node, which is computed using :attr:`values`
        """
        pass

    def visiting_potential_leaf(self, node):
        """
        Visit a node and return its value if it is a leaf.

        Note:
            This method needs to be over-written for a specific
            visitor application.

        Args:
            node: a node in a tree

        Returns:
            A tuple: ``(flag, value)``.   If ``flag`` is False,
            then the node is not a leaf and ``value`` is :const:`None`.
            Otherwise, ``value`` is the computed value for this node.
        """
        raise RuntimeError('The visiting_potential_leaf method needs to be defined.')

    def finalize(self, ans):
        """
        This method defines the return value for the search methods
        in this class.

        The default implementation returns the value of the
        initial node (aka the root node), because
        this visitor pattern computes and returns value for each
        node to enable the computation of this value.

        Args:
            ans: The final value computed by the search method.

        Returns:
            The final value after the search. Defaults to simply
            returning :attr:`ans`.
        """
        return ans

    def dfs_postorder_stack(self, node):
        """
        Perform a depth-first search in postorder using a stack
        implementation.

        Note:
            This method has the same functionality as the
            PyUtilib :class:`ValueVisitor.dfs_postorder_stack <pyutilib.misc.visitor.ValueVisitor.dfs_postorder_stack>`
            method.  The difference is that this method
            is tailored to efficiently walk Pyomo expression trees.

        Args:
            node: The root node of the expression tree
                that is searched.

        Returns:
            The return value is determined by the :func:`finalize` function,
            which may be defined by the user.
        """
        flag, value = self.visiting_potential_leaf(node)
        if flag:
            return self.finalize(value)
        _stack = [(node, node._args_, 0, node.nargs(), [])]
        while 1:
            _obj, _argList, _idx, _len, _result = _stack.pop()
            while _idx < _len:
                _sub = _argList[_idx]
                _idx += 1
                flag, value = self.visiting_potential_leaf(_sub)
                if flag:
                    _result.append(value)
                else:
                    _stack.append((_obj, _argList, _idx, _len, _result))
                    _obj = _sub
                    _argList = _sub._args_
                    _idx = 0
                    _len = _sub.nargs()
                    _result = []
            ans = self.visit(_obj, _result)
            if _stack:
                _stack[-1][-1].append(ans)
            else:
                return self.finalize(ans)