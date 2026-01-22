from __future__ import annotations
import ast
from functools import (
from keyword import iskeyword
import tokenize
from typing import (
import numpy as np
from pandas.errors import UndefinedVariableError
import pandas.core.common as com
from pandas.core.computation.ops import (
from pandas.core.computation.parsing import (
from pandas.core.computation.scope import Scope
from pandas.io.formats import printing
@disallow(_unsupported_nodes)
@add_ops(_op_classes)
class BaseExprVisitor(ast.NodeVisitor):
    """
    Custom ast walker. Parsers of other engines should subclass this class
    if necessary.

    Parameters
    ----------
    env : Scope
    engine : str
    parser : str
    preparser : callable
    """
    const_type: ClassVar[type[Term]] = Constant
    term_type: ClassVar[type[Term]] = Term
    binary_ops = CMP_OPS_SYMS + BOOL_OPS_SYMS + ARITH_OPS_SYMS
    binary_op_nodes = ('Gt', 'Lt', 'GtE', 'LtE', 'Eq', 'NotEq', 'In', 'NotIn', 'BitAnd', 'BitOr', 'And', 'Or', 'Add', 'Sub', 'Mult', None, 'Pow', 'FloorDiv', 'Mod')
    binary_op_nodes_map = dict(zip(binary_ops, binary_op_nodes))
    unary_ops = UNARY_OPS_SYMS
    unary_op_nodes = ('UAdd', 'USub', 'Invert', 'Not')
    unary_op_nodes_map = dict(zip(unary_ops, unary_op_nodes))
    rewrite_map = {ast.Eq: ast.In, ast.NotEq: ast.NotIn, ast.In: ast.In, ast.NotIn: ast.NotIn}
    unsupported_nodes: tuple[str, ...]

    def __init__(self, env, engine, parser, preparser=_preparse) -> None:
        self.env = env
        self.engine = engine
        self.parser = parser
        self.preparser = preparser
        self.assigner = None

    def visit(self, node, **kwargs):
        if isinstance(node, str):
            clean = self.preparser(node)
            try:
                node = ast.fix_missing_locations(ast.parse(clean))
            except SyntaxError as e:
                if any((iskeyword(x) for x in clean.split())):
                    e.msg = 'Python keyword not valid identifier in numexpr query'
                raise e
        method = f'visit_{type(node).__name__}'
        visitor = getattr(self, method)
        return visitor(node, **kwargs)

    def visit_Module(self, node, **kwargs):
        if len(node.body) != 1:
            raise SyntaxError('only a single expression is allowed')
        expr = node.body[0]
        return self.visit(expr, **kwargs)

    def visit_Expr(self, node, **kwargs):
        return self.visit(node.value, **kwargs)

    def _rewrite_membership_op(self, node, left, right):
        op_instance = node.op
        op_type = type(op_instance)
        if is_term(left) and is_term(right) and (op_type in self.rewrite_map):
            left_list, right_list = map(_is_list, (left, right))
            left_str, right_str = map(_is_str, (left, right))
            if left_list or right_list or left_str or right_str:
                op_instance = self.rewrite_map[op_type]()
            if right_str:
                name = self.env.add_tmp([right.value])
                right = self.term_type(name, self.env)
            if left_str:
                name = self.env.add_tmp([left.value])
                left = self.term_type(name, self.env)
        op = self.visit(op_instance)
        return (op, op_instance, left, right)

    def _maybe_transform_eq_ne(self, node, left=None, right=None):
        if left is None:
            left = self.visit(node.left, side='left')
        if right is None:
            right = self.visit(node.right, side='right')
        op, op_class, left, right = self._rewrite_membership_op(node, left, right)
        return (op, op_class, left, right)

    def _maybe_downcast_constants(self, left, right):
        f32 = np.dtype(np.float32)
        if left.is_scalar and hasattr(left, 'value') and (not right.is_scalar) and (right.return_type == f32):
            name = self.env.add_tmp(np.float32(left.value))
            left = self.term_type(name, self.env)
        if right.is_scalar and hasattr(right, 'value') and (not left.is_scalar) and (left.return_type == f32):
            name = self.env.add_tmp(np.float32(right.value))
            right = self.term_type(name, self.env)
        return (left, right)

    def _maybe_eval(self, binop, eval_in_python):
        return binop.evaluate(self.env, self.engine, self.parser, self.term_type, eval_in_python)

    def _maybe_evaluate_binop(self, op, op_class, lhs, rhs, eval_in_python=('in', 'not in'), maybe_eval_in_python=('==', '!=', '<', '>', '<=', '>=')):
        res = op(lhs, rhs)
        if res.has_invalid_return_type:
            raise TypeError(f"unsupported operand type(s) for {res.op}: '{lhs.type}' and '{rhs.type}'")
        if self.engine != 'pytables' and (res.op in CMP_OPS_SYMS and getattr(lhs, 'is_datetime', False) or getattr(rhs, 'is_datetime', False)):
            return self._maybe_eval(res, self.binary_ops)
        if res.op in eval_in_python:
            return self._maybe_eval(res, eval_in_python)
        elif self.engine != 'pytables':
            if getattr(lhs, 'return_type', None) == object or getattr(rhs, 'return_type', None) == object:
                return self._maybe_eval(res, eval_in_python + maybe_eval_in_python)
        return res

    def visit_BinOp(self, node, **kwargs):
        op, op_class, left, right = self._maybe_transform_eq_ne(node)
        left, right = self._maybe_downcast_constants(left, right)
        return self._maybe_evaluate_binop(op, op_class, left, right)

    def visit_Div(self, node, **kwargs):
        return lambda lhs, rhs: Div(lhs, rhs)

    def visit_UnaryOp(self, node, **kwargs):
        op = self.visit(node.op)
        operand = self.visit(node.operand)
        return op(operand)

    def visit_Name(self, node, **kwargs) -> Term:
        return self.term_type(node.id, self.env, **kwargs)

    def visit_NameConstant(self, node, **kwargs) -> Term:
        return self.const_type(node.value, self.env)

    def visit_Num(self, node, **kwargs) -> Term:
        return self.const_type(node.value, self.env)

    def visit_Constant(self, node, **kwargs) -> Term:
        return self.const_type(node.value, self.env)

    def visit_Str(self, node, **kwargs) -> Term:
        name = self.env.add_tmp(node.s)
        return self.term_type(name, self.env)

    def visit_List(self, node, **kwargs) -> Term:
        name = self.env.add_tmp([self.visit(e)(self.env) for e in node.elts])
        return self.term_type(name, self.env)
    visit_Tuple = visit_List

    def visit_Index(self, node, **kwargs):
        """df.index[4]"""
        return self.visit(node.value)

    def visit_Subscript(self, node, **kwargs) -> Term:
        from pandas import eval as pd_eval
        value = self.visit(node.value)
        slobj = self.visit(node.slice)
        result = pd_eval(slobj, local_dict=self.env, engine=self.engine, parser=self.parser)
        try:
            v = value.value[result]
        except AttributeError:
            lhs = pd_eval(value, local_dict=self.env, engine=self.engine, parser=self.parser)
            v = lhs[result]
        name = self.env.add_tmp(v)
        return self.term_type(name, env=self.env)

    def visit_Slice(self, node, **kwargs) -> slice:
        """df.index[slice(4,6)]"""
        lower = node.lower
        if lower is not None:
            lower = self.visit(lower).value
        upper = node.upper
        if upper is not None:
            upper = self.visit(upper).value
        step = node.step
        if step is not None:
            step = self.visit(step).value
        return slice(lower, upper, step)

    def visit_Assign(self, node, **kwargs):
        """
        support a single assignment node, like

        c = a + b

        set the assigner at the top level, must be a Name node which
        might or might not exist in the resolvers

        """
        if len(node.targets) != 1:
            raise SyntaxError('can only assign a single expression')
        if not isinstance(node.targets[0], ast.Name):
            raise SyntaxError('left hand side of an assignment must be a single name')
        if self.env.target is None:
            raise ValueError('cannot assign without a target object')
        try:
            assigner = self.visit(node.targets[0], **kwargs)
        except UndefinedVariableError:
            assigner = node.targets[0].id
        self.assigner = getattr(assigner, 'name', assigner)
        if self.assigner is None:
            raise SyntaxError('left hand side of an assignment must be a single resolvable name')
        return self.visit(node.value, **kwargs)

    def visit_Attribute(self, node, **kwargs):
        attr = node.attr
        value = node.value
        ctx = node.ctx
        if isinstance(ctx, ast.Load):
            resolved = self.visit(value).value
            try:
                v = getattr(resolved, attr)
                name = self.env.add_tmp(v)
                return self.term_type(name, self.env)
            except AttributeError:
                if isinstance(value, ast.Name) and value.id == attr:
                    return resolved
                raise
        raise ValueError(f'Invalid Attribute context {type(ctx).__name__}')

    def visit_Call(self, node, side=None, **kwargs):
        if isinstance(node.func, ast.Attribute) and node.func.attr != '__call__':
            res = self.visit_Attribute(node.func)
        elif not isinstance(node.func, ast.Name):
            raise TypeError('Only named functions are supported')
        else:
            try:
                res = self.visit(node.func)
            except UndefinedVariableError:
                try:
                    res = FuncNode(node.func.id)
                except ValueError:
                    raise
        if res is None:
            raise ValueError(f'Invalid function call {node.func.id}')
        if hasattr(res, 'value'):
            res = res.value
        if isinstance(res, FuncNode):
            new_args = [self.visit(arg) for arg in node.args]
            if node.keywords:
                raise TypeError(f'Function "{res.name}" does not support keyword arguments')
            return res(*new_args)
        else:
            new_args = [self.visit(arg)(self.env) for arg in node.args]
            for key in node.keywords:
                if not isinstance(key, ast.keyword):
                    raise ValueError(f"keyword error in function call '{node.func.id}'")
                if key.arg:
                    kwargs[key.arg] = self.visit(key.value)(self.env)
            name = self.env.add_tmp(res(*new_args, **kwargs))
            return self.term_type(name=name, env=self.env)

    def translate_In(self, op):
        return op

    def visit_Compare(self, node, **kwargs):
        ops = node.ops
        comps = node.comparators
        if len(comps) == 1:
            op = self.translate_In(ops[0])
            binop = ast.BinOp(op=op, left=node.left, right=comps[0])
            return self.visit(binop)
        left = node.left
        values = []
        for op, comp in zip(ops, comps):
            new_node = self.visit(ast.Compare(comparators=[comp], left=left, ops=[self.translate_In(op)]))
            left = comp
            values.append(new_node)
        return self.visit(ast.BoolOp(op=ast.And(), values=values))

    def _try_visit_binop(self, bop):
        if isinstance(bop, (Op, Term)):
            return bop
        return self.visit(bop)

    def visit_BoolOp(self, node, **kwargs):

        def visitor(x, y):
            lhs = self._try_visit_binop(x)
            rhs = self._try_visit_binop(y)
            op, op_class, lhs, rhs = self._maybe_transform_eq_ne(node, lhs, rhs)
            return self._maybe_evaluate_binop(op, node.op, lhs, rhs)
        operands = node.values
        return reduce(visitor, operands)