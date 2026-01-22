import os
import platform
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import native_types, nonpyomo_leaf_types, NumericConstant
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import (
from pyomo.core.base.param import _ParamData, ScalarParam
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.core.expr.compare import assertExpressionsEqual
class BaseStreamBasedVisitorTests(object):

    def setUp(self):
        self.m = m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        self.e = SumExpression([PowExpression((m.x, 2)), m.y, ProductExpression((m.z, SumExpression([m.x, m.y])))])

    def test_bad_args(self):
        with self.assertRaisesRegex(RuntimeError, "Unrecognized keyword arguments: {'foo': None}"):
            StreamBasedExpressionVisitor(foo=None)

    def test_default(self):
        walker = StreamBasedExpressionVisitor()
        ans = self.walk(walker, self.e)
        ref = [[[], []], [], [[], [[], []]]]
        self.assertEqual(ans, ref)

    def test_beforeChild(self):

        def before(node, child, child_idx):
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return (False, [child])
        walker = StreamBasedExpressionVisitor(beforeChild=before)
        ans = self.walk(walker, self.e)
        m = self.m
        ref = [[[m.x], [2]], [m.y], [[m.z], [[m.x], [m.y]]]]
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, m.x)
        ref = []
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, 2)
        ref = []
        self.assertEqual(str(ans), str(ref))

    def test_initializeWalker_beforeChild(self):

        def before(node, child, child_idx):
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return (False, child)

        def initialize(expr):
            ans = before(None, expr, 0)
            if ans is None:
                return (True, expr)
            else:
                return ans
        walker = StreamBasedExpressionVisitor(beforeChild=before, initializeWalker=initialize)
        ans = self.walk(walker, self.e)
        m = self.m
        ref = [[m.x, 2], m.y, [m.z, [m.x, m.y]]]
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, m.x)
        ref = m.x
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, 2)
        ref = 2
        self.assertEqual(str(ans), str(ref))

    def test_beforeChild_exitNode(self):

        def before(node, child, child_idx):
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return (False, [child])

        def exit(node, data):
            if hasattr(node, 'getname'):
                data.insert(0, node.getname())
            else:
                data.insert(0, str(node))
            return data
        walker = StreamBasedExpressionVisitor(beforeChild=before, exitNode=exit)
        ans = self.walk(walker, self.e)
        m = self.m
        ref = ['sum', ['pow', [m.x], [2]], [m.y], ['prod', [m.z], ['sum', [m.x], [m.y]]]]
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, m.x)
        ref = ['x']
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, 2)
        ref = ['2']
        self.assertEqual(str(ans), str(ref))

    def test_beforeChild_enterNode_exitNode(self):
        i = [0]

        def before(node, child, child_idx):
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return (False, [child])

        def enter(node):
            i[0] += 1
            return (None, [i[0]])

        def exit(node, data):
            if hasattr(node, 'getname'):
                data.insert(0, node.getname())
            else:
                data.insert(0, str(node))
            return data
        walker = StreamBasedExpressionVisitor(beforeChild=before, enterNode=enter, exitNode=exit)
        ans = self.walk(walker, self.e)
        m = self.m
        ref = ['sum', 1, ['pow', 2, [m.x], [2]], [m.y], ['prod', 3, [m.z], ['sum', 4, [m.x], [m.y]]]]
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, m.x)
        ref = ['x', 5]
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, 2)
        ref = ['2', 6]
        self.assertEqual(str(ans), str(ref))

    def test_old_beforeChild(self):

        def before(node, child):
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return (False, [child])
        os = StringIO()
        with LoggingIntercept(os, 'pyomo'):
            walker = StreamBasedExpressionVisitor(beforeChild=before)
        self.assertIn('Note that the API for the StreamBasedExpressionVisitor has changed to include the child index for the beforeChild() method', os.getvalue().replace('\n', ' '))
        ans = self.walk(walker, self.e)
        m = self.m
        ref = [[[m.x], [2]], [m.y], [[m.z], [[m.x], [m.y]]]]
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, m.x)
        ref = []
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, 2)
        ref = []
        self.assertEqual(str(ans), str(ref))

    def test_reduce_in_accept(self):

        def enter(node):
            return (None, 1)

        def accept(node, data, child_result, child_idx):
            return data + child_result
        walker = StreamBasedExpressionVisitor(enterNode=enter, acceptChildResult=accept)
        self.assertEqual(self.walk(walker, self.e), 10)

    def test_sizeof_expression(self):
        self.assertEqual(sizeof_expression(self.e), 10)

    def test_enterNode(self):

        def enter(node):
            if type(node) in nonpyomo_leaf_types or not node.is_expression_type():
                return ((), [node])
            return (node.args, [])
        walker = StreamBasedExpressionVisitor(enterNode=enter)
        m = self.m
        ans = self.walk(walker, self.e)
        ref = [[[m.x], [2]], [m.y], [[m.z], [[m.x], [m.y]]]]
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, m.x)
        ref = [m.x]
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, 2)
        ref = [2]
        self.assertEqual(str(ans), str(ref))

    def test_enterNode_noLeafList(self):

        def enter(node):
            if type(node) in nonpyomo_leaf_types or not node.is_expression_type():
                return ((), node)
            return (node.args, [])
        walker = StreamBasedExpressionVisitor(enterNode=enter)
        m = self.m
        ans = self.walk(walker, self.e)
        ref = [[m.x, 2], m.y, [m.z, [m.x, m.y]]]
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, m.x)
        ref = m.x
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, 2)
        ref = 2
        self.assertEqual(str(ans), str(ref))

    def test_enterNode_withFinalize(self):

        def enter(node):
            if type(node) in nonpyomo_leaf_types or not node.is_expression_type():
                return ((), node)
            return (node.args, [])

        def finalize(result):
            if type(result) is list:
                return result
            else:
                return [result]
        walker = StreamBasedExpressionVisitor(enterNode=enter, finalizeResult=finalize)
        m = self.m
        ans = self.walk(walker, self.e)
        ref = [[m.x, 2], m.y, [m.z, [m.x, m.y]]]
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, m.x)
        ref = [m.x]
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, 2)
        ref = [2]
        self.assertEqual(str(ans), str(ref))

    def test_exitNode(self):

        def exit(node, data):
            if data:
                return data
            else:
                return [node]
        walker = StreamBasedExpressionVisitor(exitNode=exit)
        m = self.m
        ans = self.walk(walker, self.e)
        ref = [[[m.x], [2]], [m.y], [[m.z], [[m.x], [m.y]]]]
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, m.x)
        ref = [m.x]
        self.assertEqual(str(ans), str(ref))
        ans = self.walk(walker, 2)
        ref = [2]
        self.assertEqual(str(ans), str(ref))

    def test_beforeChild_acceptChildResult_afterChild(self):
        counts = [0, 0, 0]

        def before(node, child, child_idx):
            counts[0] += 1
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return (False, None)

        def accept(node, data, child_result, child_idx):
            counts[1] += 1

        def after(node, child, child_idx):
            counts[2] += 1
        walker = StreamBasedExpressionVisitor(beforeChild=before, acceptChildResult=accept, afterChild=after)
        ans = self.walk(walker, self.e)
        m = self.m
        self.assertEqual(ans, None)
        self.assertEqual(counts, [9, 9, 9])

    def test_OLD_beforeChild_acceptChildResult_afterChild(self):
        counts = [0, 0, 0]

        def before(node, child):
            counts[0] += 1
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return (False, None)

        def accept(node, data, child_result):
            counts[1] += 1

        def after(node, child):
            counts[2] += 1
        os = StringIO()
        with LoggingIntercept(os, 'pyomo'):
            walker = StreamBasedExpressionVisitor(beforeChild=before, acceptChildResult=accept, afterChild=after)
        self.assertIn('Note that the API for the StreamBasedExpressionVisitor has changed to include the child index for the beforeChild() method', os.getvalue().replace('\n', ' '))
        self.assertIn('Note that the API for the StreamBasedExpressionVisitor has changed to include the child index for the acceptChildResult() method', os.getvalue().replace('\n', ' '))
        self.assertIn('Note that the API for the StreamBasedExpressionVisitor has changed to include the child index for the afterChild() method', os.getvalue().replace('\n', ' '))
        ans = self.walk(walker, self.e)
        m = self.m
        self.assertEqual(ans, None)
        self.assertEqual(counts, [9, 9, 9])

    def test_enterNode_acceptChildResult_beforeChild(self):
        ans = []

        def before(node, child, child_idx):
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return (False, child)

        def accept(node, data, child_result, child_idx):
            if data is not child_result:
                data.append(child_result)
            return data

        def enter(node):
            return (node.args, ans)
        walker = StreamBasedExpressionVisitor(enterNode=enter, beforeChild=before, acceptChildResult=accept)
        ans = self.walk(walker, self.e)
        m = self.m
        ref = [m.x, 2, m.y, m.z, m.x, m.y]
        self.assertEqual(str(ans), str(ref))

    def test_finalize(self):
        ans = []

        def before(node, child, child_idx):
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return (False, child)

        def accept(node, data, child_result, child_idx):
            if data is not child_result:
                data.append(child_result)
            return data

        def enter(node):
            return (node.args, ans)

        def finalize(result):
            return len(result)
        walker = StreamBasedExpressionVisitor(enterNode=enter, beforeChild=before, acceptChildResult=accept, finalizeResult=finalize)
        ans = self.walk(walker, self.e)
        self.assertEqual(ans, 6)

    def test_all_function_pointers(self):
        ans = []

        def name(x):
            if type(x) in nonpyomo_leaf_types:
                return str(x)
            else:
                return x.name

        def initialize(expr):
            ans.append('Initialize')
            return (True, None)

        def enter(node):
            ans.append('Enter %s' % name(node))

        def exit(node, data):
            ans.append('Exit %s' % name(node))

        def before(node, child, child_idx):
            ans.append('Before %s (from %s)' % (name(child), name(node)))

        def accept(node, data, child_result, child_idx):
            ans.append('Accept into %s' % name(node))

        def after(node, child, child_idx):
            ans.append('After %s (from %s)' % (name(child), name(node)))

        def finalize(result):
            ans.append('Finalize')
        walker = StreamBasedExpressionVisitor(initializeWalker=initialize, enterNode=enter, exitNode=exit, beforeChild=before, acceptChildResult=accept, afterChild=after, finalizeResult=finalize)
        self.assertIsNone(self.walk(walker, self.e))
        self.assertEqual('\n'.join(ans), 'Initialize\nEnter sum\nBefore pow (from sum)\nEnter pow\nBefore x (from pow)\nEnter x\nExit x\nAccept into pow\nAfter x (from pow)\nBefore 2 (from pow)\nEnter 2\nExit 2\nAccept into pow\nAfter 2 (from pow)\nExit pow\nAccept into sum\nAfter pow (from sum)\nBefore y (from sum)\nEnter y\nExit y\nAccept into sum\nAfter y (from sum)\nBefore prod (from sum)\nEnter prod\nBefore z (from prod)\nEnter z\nExit z\nAccept into prod\nAfter z (from prod)\nBefore sum (from prod)\nEnter sum\nBefore x (from sum)\nEnter x\nExit x\nAccept into sum\nAfter x (from sum)\nBefore y (from sum)\nEnter y\nExit y\nAccept into sum\nAfter y (from sum)\nExit sum\nAccept into prod\nAfter sum (from prod)\nExit prod\nAccept into sum\nAfter prod (from sum)\nExit sum\nFinalize')

    def test_all_derived_class(self):

        def name(x):
            if type(x) in nonpyomo_leaf_types:
                return str(x)
            else:
                return x.name

        class all_callbacks(StreamBasedExpressionVisitor):

            def __init__(self):
                self.ans = []
                super(all_callbacks, self).__init__()

            def initializeWalker(self, expr):
                self.ans.append('Initialize')
                return (True, None)

            def enterNode(self, node):
                self.ans.append('Enter %s' % name(node))

            def exitNode(self, node, data):
                self.ans.append('Exit %s' % name(node))

            def beforeChild(self, node, child, child_idx):
                self.ans.append('Before %s (from %s)' % (name(child), name(node)))

            def acceptChildResult(self, node, data, child_result, child_idx):
                self.ans.append('Accept into %s' % name(node))

            def afterChild(self, node, child, child_idx):
                self.ans.append('After %s (from %s)' % (name(child), name(node)))

            def finalizeResult(self, result):
                self.ans.append('Finalize')
        walker = all_callbacks()
        self.assertIsNone(self.walk(walker, self.e))
        self.assertEqual('\n'.join(walker.ans), 'Initialize\nEnter sum\nBefore pow (from sum)\nEnter pow\nBefore x (from pow)\nEnter x\nExit x\nAccept into pow\nAfter x (from pow)\nBefore 2 (from pow)\nEnter 2\nExit 2\nAccept into pow\nAfter 2 (from pow)\nExit pow\nAccept into sum\nAfter pow (from sum)\nBefore y (from sum)\nEnter y\nExit y\nAccept into sum\nAfter y (from sum)\nBefore prod (from sum)\nEnter prod\nBefore z (from prod)\nEnter z\nExit z\nAccept into prod\nAfter z (from prod)\nBefore sum (from prod)\nEnter sum\nBefore x (from sum)\nEnter x\nExit x\nAccept into sum\nAfter x (from sum)\nBefore y (from sum)\nEnter y\nExit y\nAccept into sum\nAfter y (from sum)\nExit sum\nAccept into prod\nAfter sum (from prod)\nExit prod\nAccept into sum\nAfter prod (from sum)\nExit sum\nFinalize')

    def test_all_derived_class_oldAPI(self):

        def name(x):
            if type(x) in nonpyomo_leaf_types:
                return str(x)
            else:
                return x.name

        class all_callbacks(StreamBasedExpressionVisitor):

            def __init__(self):
                self.ans = []
                super(all_callbacks, self).__init__()

            def enterNode(self, node):
                self.ans.append('Enter %s' % name(node))

            def exitNode(self, node, data):
                self.ans.append('Exit %s' % name(node))

            def beforeChild(self, node, child):
                self.ans.append('Before %s (from %s)' % (name(child), name(node)))

            def acceptChildResult(self, node, data, child_result):
                self.ans.append('Accept into %s' % name(node))

            def afterChild(self, node, child):
                self.ans.append('After %s (from %s)' % (name(child), name(node)))

            def finalizeResult(self, result):
                self.ans.append('Finalize')
        os = StringIO()
        with LoggingIntercept(os, 'pyomo'):
            walker = all_callbacks()
        self.assertIn('Note that the API for the StreamBasedExpressionVisitor has changed to include the child index for the beforeChild() method', os.getvalue().replace('\n', ' '))
        self.assertIn('Note that the API for the StreamBasedExpressionVisitor has changed to include the child index for the acceptChildResult() method', os.getvalue().replace('\n', ' '))
        self.assertIn('Note that the API for the StreamBasedExpressionVisitor has changed to include the child index for the afterChild() method', os.getvalue().replace('\n', ' '))
        self.assertIsNone(self.walk(walker, self.e))
        self.assertEqual('\n'.join(walker.ans), 'Enter sum\nBefore pow (from sum)\nEnter pow\nBefore x (from pow)\nEnter x\nExit x\nAccept into pow\nAfter x (from pow)\nBefore 2 (from pow)\nEnter 2\nExit 2\nAccept into pow\nAfter 2 (from pow)\nExit pow\nAccept into sum\nAfter pow (from sum)\nBefore y (from sum)\nEnter y\nExit y\nAccept into sum\nAfter y (from sum)\nBefore prod (from sum)\nEnter prod\nBefore z (from prod)\nEnter z\nExit z\nAccept into prod\nAfter z (from prod)\nBefore sum (from prod)\nEnter sum\nBefore x (from sum)\nEnter x\nExit x\nAccept into sum\nAfter x (from sum)\nBefore y (from sum)\nEnter y\nExit y\nAccept into sum\nAfter y (from sum)\nExit sum\nAccept into prod\nAfter sum (from prod)\nExit prod\nAccept into sum\nAfter prod (from sum)\nExit sum\nFinalize')