import ast
from collections import defaultdict, OrderedDict
import contextlib
import sys
from types import SimpleNamespace
import numpy as np
import operator
from numba.core import types, targetconfig, ir, rewrites, compiler
from numba.core.typing import npydecl
from numba.np.ufunc.dufunc import DUFunc
def _arr_expr_to_ast(expr):
    """Build a Python expression AST from an array expression built by
    RewriteArrayExprs.
    """
    if isinstance(expr, tuple):
        op, arr_expr_args = expr
        ast_args = []
        env = {}
        for arg in arr_expr_args:
            ast_arg, child_env = _arr_expr_to_ast(arg)
            ast_args.append(ast_arg)
            env.update(child_env)
        if op in npydecl.supported_array_operators:
            if len(ast_args) == 2:
                if op in _binops:
                    return (ast.BinOp(ast_args[0], _binops[op](), ast_args[1]), env)
                if op in _cmpops:
                    return (ast.Compare(ast_args[0], [_cmpops[op]()], [ast_args[1]]), env)
            else:
                assert op in _unaryops
                return (ast.UnaryOp(_unaryops[op](), ast_args[0]), env)
        elif _is_ufunc(op):
            fn_name = '__ufunc_or_dufunc_{0}'.format(hex(hash(op)).replace('-', '_'))
            fn_ast_name = ast.Name(fn_name, ast.Load())
            env[fn_name] = op
            ast_call = ast.Call(fn_ast_name, ast_args, [])
            return (ast_call, env)
    elif isinstance(expr, ir.Var):
        return (ast.Name(expr.name, ast.Load(), lineno=expr.loc.line, col_offset=expr.loc.col if expr.loc.col else 0), {})
    elif isinstance(expr, ir.Const):
        return (ast.Constant(expr.value), {})
    raise NotImplementedError("Don't know how to translate array expression '%r'" % (expr,))