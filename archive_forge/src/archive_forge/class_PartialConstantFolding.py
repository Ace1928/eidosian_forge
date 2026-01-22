from pythran.analyses import ConstantExpressions, ASTMatcher
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.conversion import to_ast, ConversionError, ToNotEval, mangle
from pythran.analyses.ast_matcher import DamnTooLongPattern
from pythran.syntax import PythranSyntaxError
from pythran.utils import isintegral, isnum
from pythran.config import cfg
import builtins
import gast as ast
from copy import deepcopy
import logging
import sys
class PartialConstantFolding(Transformation):
    """
    Replace partially constant expression by their evaluation.

    >>> import gast as ast
    >>> from pythran import passmanager, backend

    >>> node = ast.parse("def foo(n): return [n] * 2")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(PartialConstantFolding, node)
    >>> print(pm.dump(backend.Python, node))
    def foo(n):
        return [n, n]

    >>> node = ast.parse("def foo(n): return 2 * (n,)")
    >>> _, node = pm.apply(PartialConstantFolding, node)
    >>> print(pm.dump(backend.Python, node))
    def foo(n):
        return (n, n)

    >>> node = ast.parse("def foo(n): return [n] + [n]")
    >>> _, node = pm.apply(PartialConstantFolding, node)
    >>> print(pm.dump(backend.Python, node))
    def foo(n):
        return [n, n]

    >>> node = ast.parse("def foo(n, m): return (n,) + (m, n)")
    >>> _, node = pm.apply(PartialConstantFolding, node)
    >>> print(pm.dump(backend.Python, node))
    def foo(n, m):
        return (n, m, n)
    """

    def __init__(self):
        Transformation.__init__(self, ConstantExpressions)

    def fold_mult_left(self, node):
        if not isinstance(node.left, (ast.List, ast.Tuple)):
            return False
        if not isnum(node.right):
            return False
        if not isintegral(node.right):
            raise PythranSyntaxError('Multiplying a sequence by a float', node)
        return isinstance(node.op, ast.Mult)

    def fold_mult_right(self, node):
        if not isinstance(node.right, (ast.List, ast.Tuple)):
            return False
        if not isnum(node.left):
            return False
        if not isintegral(node.left):
            raise PythranSyntaxError('Multiplying a sequence by a float', node)
        return isinstance(node.op, ast.Mult)

    def fold_add(self, node, ty):
        if not isinstance(node.left, ty):
            return False
        if not isinstance(node.right, ty):
            return False
        return isinstance(node.op, ast.Add)

    def visit_BinOp(self, node):
        if node in self.constant_expressions:
            return node
        node = self.generic_visit(node)
        if self.fold_mult_left(node):
            self.update = True
            node.left.elts = [deepcopy(elt) for _ in range(node.right.value) for elt in node.left.elts]
            return node.left
        if self.fold_mult_right(node):
            self.update = True
            node.left, node.right = (node.right, node.left)
            return self.visit(node)
        for ty in (ast.List, ast.Tuple):
            if self.fold_add(node, ty):
                self.update = True
                node.left.elts += node.right.elts
                return node.left
        return node

    def visit_Subscript(self, node):
        """
        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> pm = passmanager.PassManager("test")

        >>> node = ast.parse("def foo(a): a[1:][3]")
        >>> _, node = pm.apply(PartialConstantFolding, node)
        >>> _, node = pm.apply(ConstantFolding, node)
        >>> print(pm.dump(backend.Python, node))
        def foo(a):
            a[4]

        >>> node = ast.parse("def foo(a): a[::2][3]")
        >>> _, node = pm.apply(PartialConstantFolding, node)
        >>> _, node = pm.apply(ConstantFolding, node)
        >>> print(pm.dump(backend.Python, node))
        def foo(a):
            a[6]

        >>> node = ast.parse("def foo(a): a[-4:][5]")
        >>> _, node = pm.apply(PartialConstantFolding, node)
        >>> _, node = pm.apply(ConstantFolding, node)
        >>> print(pm.dump(backend.Python, node))
        def foo(a):
            a[1]
        """
        self.generic_visit(node)
        if not isinstance(node.value, ast.Subscript):
            return node
        if not isinstance(node.value.slice, ast.Slice):
            return node
        if not isintegral(node.slice):
            return node
        slice_ = node.value.slice
        index = node.slice
        node = node.value
        lower = slice_.lower or ast.Constant(0, None)
        step = slice_.step or ast.Constant(1, None)
        node.slice = ast.BinOp(lower, ast.Add(), ast.BinOp(index, ast.Mult(), step))
        self.update = True
        return node