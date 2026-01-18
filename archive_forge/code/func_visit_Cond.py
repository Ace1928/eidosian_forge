from pythran.analyses import (ImportedIds, HasReturn, IsAssigned, CFG,
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy
def visit_Cond(self, node):
    """
        generic expression splitting algorithm. Should work for ifexp and if
        using W(rap) and U(n)W(rap) to manage difference between expr and stmt

        The idea is to split a BinOp in three expressions:
            1. a (possibly empty) non-static expr
            2. an expr containing a static expr
            3. a (possibly empty) non-static expr
        Once split, the if body is refactored to keep the semantic,
        and then recursively split again, until all static expr are alone in a
        test condition
        """
    NodeTy = type(node)
    if NodeTy is ast.IfExp:

        def W(x):
            return x

        def UW(x):
            return x
    else:

        def W(x):
            return [x]

        def UW(x):
            return x[0]
    has_static_expr = self.gather(HasStaticExpression, node.test)
    if not has_static_expr:
        return self.generic_visit(node)
    if node.test in self.static_expressions:
        return self.generic_visit(node)
    if not isinstance(node.test, ast.BinOp):
        return self.generic_visit(node)
    before, static = ([], [])
    values = [node.test.right, node.test.left]

    def has_static_expression(n):
        return self.gather(HasStaticExpression, n)
    while values and (not has_static_expression(values[-1])):
        before.append(values.pop())
    while values and has_static_expression(values[-1]):
        static.append(values.pop())
    after = list(reversed(values))
    test_before = NodeTy(None, None, None)
    if before:
        assert len(before) == 1
        test_before.test = before[0]
    test_static = NodeTy(None, None, None)
    if static:
        test_static.test = static[0]
        if len(static) > 1:
            if after:
                assert len(after) == 1
                after = [ast.BinOp(static[1], node.test.op, after[0])]
            else:
                after = static[1:]
    test_after = NodeTy(None, None, None)
    if after:
        assert len(after) == 1
        test_after.test = after[0]
    if isinstance(node.test.op, ast.BitAnd):
        if after:
            test_after.body = deepcopy(node.body)
            test_after.orelse = deepcopy(node.orelse)
            test_after = W(test_after)
        else:
            test_after = deepcopy(node.body)
        if static:
            test_static.body = test_after
            test_static.orelse = deepcopy(node.orelse)
            test_static = W(test_static)
        else:
            test_static = test_after
        if before:
            test_before.body = test_static
            test_before.orelse = node.orelse
            node = test_before
        else:
            node = UW(test_static)
    elif isinstance(node.test.op, ast.BitOr):
        if after:
            test_after.body = deepcopy(node.body)
            test_after.orelse = deepcopy(node.orelse)
            test_after = W(test_after)
        else:
            test_after = deepcopy(node.orelse)
        if static:
            test_static.body = deepcopy(node.body)
            test_static.orelse = test_after
            test_static = W(test_static)
        else:
            test_static = test_after
        if before:
            test_before.body = deepcopy(node.body)
            test_before.orelse = test_static
            node = test_before
        else:
            node = UW(test_static)
    else:
        raise PythranSyntaxError('operator not supported in a static if', node)
    self.update = True
    return self.generic_visit(node)