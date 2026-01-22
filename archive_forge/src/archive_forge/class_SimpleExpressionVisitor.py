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
class SimpleExpressionVisitor(object):
    """
    Note:
        This class is a customization of the PyUtilib :class:`SimpleVisitor
        <pyutilib.misc.visitor.SimpleVisitor>` class that is tailored
        to efficiently walk Pyomo expression trees.  However, this class
        is not a subclass of the PyUtilib :class:`SimpleVisitor
        <pyutilib.misc.visitor.SimpleVisitor>` class because all key methods
        are reimplemented.
    """

    def visit(self, node):
        """
        Visit a node in an expression tree and perform some operation on
        it.

        This method should be over-written by a user
        that is creating a sub-class.

        Args:
            node: a node in an expression tree

        Returns:
            nothing
        """
        pass

    def finalize(self):
        """
        Return the "final value" of the search.

        The default implementation returns :const:`None`, because
        the traditional visitor pattern does not return a value.

        Returns:
            The final value after the search.  Default is :const:`None`.
        """
        pass

    def xbfs(self, node):
        """
        Breadth-first search of an expression tree,
        except that leaf nodes are immediately visited.

        Note:
            This method has the same functionality as the
            PyUtilib :class:`SimpleVisitor.xbfs <pyutilib.misc.visitor.SimpleVisitor.xbfs>`
            method.  The difference is that this method
            is tailored to efficiently walk Pyomo expression trees.

        Args:
            node: The root node of the expression tree that is searched.

        Returns:
            The return value is determined by the :func:`finalize` function,
            which may be defined by the user.  Defaults to :const:`None`.
        """
        dq = deque([node])
        while dq:
            current = dq.popleft()
            self.visit(current)
            for c in current.args:
                if c.__class__ in nonpyomo_leaf_types or not c.is_expression_type() or c.nargs() == 0:
                    self.visit(c)
                else:
                    dq.append(c)
        return self.finalize()

    def xbfs_yield_leaves(self, node):
        """
        Breadth-first search of an expression tree, except that
        leaf nodes are immediately visited.

        Note:
            This method has the same functionality as the
            PyUtilib :class:`SimpleVisitor.xbfs_yield_leaves <pyutilib.misc.visitor.SimpleVisitor.xbfs_yield_leaves>`
            method.  The difference is that this method
            is tailored to efficiently walk Pyomo expression trees.

        Args:
            node: The root node of the expression tree
                that is searched.

        Returns:
            The return value is determined by the :func:`finalize` function,
            which may be defined by the user.  Defaults to :const:`None`.
        """
        if node.__class__ in nonpyomo_leaf_types or not node.is_expression_type() or node.nargs() == 0:
            ans = self.visit(node)
            if not ans is None:
                yield ans
            return
        dq = deque([node])
        while dq:
            current = dq.popleft()
            for c in current.args:
                if c.__class__ in nonpyomo_leaf_types or not c.is_expression_type() or c.nargs() == 0:
                    ans = self.visit(c)
                    if not ans is None:
                        yield ans
                else:
                    dq.append(c)