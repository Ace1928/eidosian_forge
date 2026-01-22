import gast as ast
from collections import defaultdict
from functools import reduce
from pythran.analyses import Aliases, CFG
from pythran.intrinsic import Intrinsic
from pythran.passmanager import ModuleAnalysis
from pythran.interval import Interval, IntervalTuple, UNKNOWN_RANGE
from pythran.tables import MODULES, attributes
class RangeValues(RangeValuesBase):
    """
    This analyse extract positive subscripts from code.

    It is flow sensitive and aliasing is not taken into account as integer
    doesn't create aliasing in Python.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse('''
    ... def foo(a):
    ...     for i in builtins.range(1, 10):
    ...         c = i // 2
    ...     return''')
    >>> pm = passmanager.PassManager("test")
    >>> res = pm.gather(RangeValues, node)
    >>> res['c'], res['i']
    (Interval(low=0, high=5), Interval(low=1, high=10))
    """

    def __init__(self):
        super(RangeValues, self).__init__()

    def generic_visit(self, node):
        """ Other nodes are not known and range value neither. """
        super(RangeValues, self).generic_visit(node)
        if isinstance(node, ast.stmt):
            if node in self.cfg:
                return self.cfg.successors(node)
        else:
            return self.add(node, UNKNOWN_RANGE)

    def cfg_visit(self, node, skip=None):
        successors = [node]
        visited = set() if skip is None else skip.copy()
        while successors:
            successor = successors.pop()
            if successor in visited:
                continue
            visited.add(successor)
            nexts = self.visit(successor)
            if nexts:
                successors.extend((n for n in nexts if n is not CFG.NIL))

    def save_state(self):
        return (self.cfg, self.aliases, self.use_omp, self.no_backward, self.no_if_split, self.result.copy())

    def restore_state(self, state):
        self.cfg, self.aliases, self.use_omp, self.no_backward, self.no_if_split, self.result = state

    def function_visitor(self, node):
        parent_result = self.result
        self.result = defaultdict(lambda: UNKNOWN_RANGE, [(k, v) for k, v in parent_result.items() if isinstance(k, ast.FunctionDef)])
        try:
            self.no_backward = 0
            self.no_if_split = 0
            self.cfg_visit(next(self.cfg.successors(node)))
            for k, v in self.result.items():
                parent_result[k] = v
            self.result = parent_result
        except RangeValueTooCostly:
            self.result = parent_result
            rvs = RangeValuesSimple(self)
            rvs.visit(node)

    def visit_Return(self, node):
        if node.value:
            return_range = self.visit(node.value)
            self.add(RangeValues.ResultHolder, return_range)
        return self.cfg.successors(node)

    def visit_Assert(self, node):
        """
        Constraint the range of variables

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse("def foo(a): assert a >= 1; b = a + 1")
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=1, high=inf)
        >>> res['b']
        Interval(low=2, high=inf)
        """
        self.visit(node.test)
        bound_range(self.result, self.aliases, node.test)
        return self.cfg.successors(node)

    def visit_Assign(self, node):
        """
        Set range value for assigned variable.

        We do not handle container values.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse("def foo(): a = b = 2")
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=2, high=2)
        >>> res['b']
        Interval(low=2, high=2)
        """
        assigned_range = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.result[target.id] = assigned_range
            else:
                self.visit(target)
        return self.cfg.successors(node)

    def visit_AugAssign(self, node):
        """ Update range value for augassigned variables.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse("def foo(): a = 2; a -= 1")
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=1, high=1)
        """
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            name = node.target.id
            res = combine(node.op, self.result[name], self.result[node.value])
            self.result[name] = res
        return self.cfg.successors(node)

    def visit_loop_successor(self, node):
        for successor in self.cfg.successors(node):
            if successor is not node.body[0]:
                if isinstance(node, ast.While):
                    bound_range(self.result, self.aliases, ast.UnaryOp(ast.Not(), node.test))
                return [successor]

    def visit_For(self, node):
        """ Handle iterate variable in for loops.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = b = c = 2
        ...     for i in builtins.range(1):
        ...         a -= 1
        ...         b += 1
        ...     return''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=-inf, high=2)
        >>> res['b']
        Interval(low=2, high=inf)
        >>> res['c']
        Interval(low=2, high=2)

        >>> node = ast.parse('''
        ... def foo():
        ...     for i in (1, 2, 4):
        ...         a = i
        ...     return''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=1, high=4)
        """
        assert isinstance(node.target, ast.Name), 'For apply on variables.'
        self.visit(node.iter)
        init_state = self.result.copy()
        bound_range(self.result, self.aliases, ast.Compare(node.target, [ast.In()], [node.iter]))
        skip = {x for x in self.cfg.successors(node) if x is not node.body[0]}
        skip.add(node)
        next_ = self.cfg_visit(node.body[0], skip=skip)
        if self.no_backward:
            return self.visit_loop_successor(node)
        else:
            pass
        prev_state = self.result
        self.result = prev_state.copy()
        self.cfg_visit(node.body[0], skip=skip)
        self.widen(self.result, prev_state)
        self.cfg_visit(node.body[0], skip=skip)
        self.unionify(init_state)
        pass
        return self.visit_loop_successor(node)

    def visit_While(self, node):
        """ Handle incremented variables in loop body.

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> node = ast.parse('''
        ... def foo():
        ...     a = b = c = 10
        ...     while a > 0:
        ...         a -= 1
        ...         b += 1
        ...     return''')
        >>> pm = passmanager.PassManager("test")
        >>> res = pm.gather(RangeValues, node)
        >>> res['a']
        Interval(low=-inf, high=0)
        >>> res['b']
        Interval(low=11, high=inf)
        >>> res['c']
        Interval(low=10, high=10)
        """
        test_range = self.visit(node.test)
        init_state = self.result.copy()
        skip = {x for x in self.cfg.successors(node) if x is not node.body[0]}
        skip.add(node)
        if 0 in test_range:
            for successor in list(self.cfg.successors(node)):
                if successor is not node.body[0]:
                    self.cfg_visit(successor, skip=skip)
        bound_range(self.result, self.aliases, node.test)
        self.cfg_visit(node.body[0], skip=skip)
        if self.no_backward:
            if 0 in test_range:
                self.unionify(init_state)
            return self.visit_loop_successor(node)
        else:
            pass
        prev_state = self.result
        self.result = prev_state.copy()
        self.cfg_visit(node.body[0], skip=skip)
        self.widen(self.result, prev_state)
        self.cfg_visit(node.body[0], skip=skip)
        if 0 in test_range:
            self.unionify(init_state)
        else:
            self.unionify(prev_state)
        self.visit(node.test)
        pass
        return self.visit_loop_successor(node)

    def visit_If(self, node):
        """ Handle iterate variable across branches

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> pm = passmanager.PassManager("test")

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if a > 1: b = 1
        ...     else: b = 3
        ...     pass''')

        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=1, high=3)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if a > 1: b = a
        ...     else: b = 3
        ...     pass''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=2, high=inf)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if 0 < a < 4: b = a
        ...     else: b = 3
        ...     pass''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=1, high=3)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if (0 < a) and (a < 4): b = a
        ...     else: b = 3
        ...     pass''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=1, high=3)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if (a == 1) or (a == 2): b = a
        ...     else: b = 3
        ...     pass''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=1, high=3)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     b = 5
        ...     if a > 0: b = a
        ...     pass''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['a'], res['b']
        (Interval(low=-inf, high=inf), Interval(low=1, high=inf))

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if a > 3: b = 1
        ...     else: b = 2
        ...     if a > 1: b = 2
        ...     pass''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=2, high=2)
        """
        if self.no_if_split == 4:
            raise RangeValueTooCostly()
        self.no_if_split += 1
        test_range = self.visit(node.test)
        init_state = self.result.copy()
        if 1 in test_range:
            bound_range(self.result, self.aliases, node.test)
            self.cfg_visit(node.body[0])
        visited_successors = {node.body[0]}
        if node.orelse:
            if 0 in test_range:
                prev_state = self.result
                self.result = init_state.copy()
                bound_range(self.result, self.aliases, ast.UnaryOp(ast.Not(), node.test))
                self.cfg_visit(node.orelse[0])
                self.unionify(prev_state)
            visited_successors.add(node.orelse[0])
        elif 0 in test_range:
            successors = self.cfg.successors(node)
            for successor in list(successors):
                if successor not in visited_successors:
                    self.result, prev_state = (init_state.copy(), self.result)
                    bound_range(self.result, self.aliases, ast.UnaryOp(ast.Not(), node.test))
                    self.cfg_visit(successor)
                    self.unionify(prev_state)
        self.no_if_split -= 1

    def visit_Try(self, node):
        init_range = self.result
        self.result = init_range.copy()
        self.cfg_visit(node.body[0])
        self.unionify(init_range)
        init_range = self.result.copy()
        for handler in node.handlers:
            self.result, prev_state = (init_range.copy(), self.result)
            self.cfg_visit(handler.body[0])
            self.unionify(prev_state)