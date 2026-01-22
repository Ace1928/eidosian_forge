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
def _lower_array_expr(lowerer, expr):
    """Lower an array expression built by RewriteArrayExprs.
    """
    expr_name = '__numba_array_expr_%s' % hex(hash(expr)).replace('-', '_')
    expr_filename = expr.loc.filename
    expr_var_list = expr.list_vars()
    expr_var_unique = sorted(set(expr_var_list), key=lambda var: var.name)
    expr_args = [var.name for var in expr_var_unique]
    with _legalize_parameter_names(expr_var_unique) as expr_params:
        ast_args = [ast.arg(param_name, None) for param_name in expr_params]
        ast_module = ast.parse('def {0}(): return'.format(expr_name), expr_filename, 'exec')
        assert hasattr(ast_module, 'body') and len(ast_module.body) == 1
        ast_fn = ast_module.body[0]
        ast_fn.args.args = ast_args
        ast_fn.body[0].value, namespace = _arr_expr_to_ast(expr.expr)
        _fix_invalid_lineno_ranges(ast_module)
    code_obj = compile(ast_module, expr_filename, 'exec')
    exec(code_obj, namespace)
    impl = namespace[expr_name]
    context = lowerer.context
    builder = lowerer.builder
    outer_sig = expr.ty(*(lowerer.typeof(name) for name in expr_args))
    inner_sig_args = []
    for argty in outer_sig.args:
        if isinstance(argty, types.Optional):
            argty = argty.type
        if isinstance(argty, types.Array):
            inner_sig_args.append(argty.dtype)
        else:
            inner_sig_args.append(argty)
    inner_sig = outer_sig.return_type.dtype(*inner_sig_args)
    flags = targetconfig.ConfigStack().top_or_none()
    flags = compiler.Flags() if flags is None else flags.copy()
    flags.error_model = 'numpy'
    cres = context.compile_subroutine(builder, impl, inner_sig, flags=flags, caching=False)
    from numba.np import npyimpl

    class ExprKernel(npyimpl._Kernel):

        def generate(self, *args):
            arg_zip = zip(args, self.outer_sig.args, inner_sig.args)
            cast_args = [self.cast(val, inty, outty) for val, inty, outty in arg_zip]
            result = self.context.call_internal(builder, cres.fndesc, inner_sig, cast_args)
            return self.cast(result, inner_sig.return_type, self.outer_sig.return_type)
    ufunc = SimpleNamespace(nin=len(expr_args), nout=1, __name__=expr_name)
    ufunc.nargs = ufunc.nin + ufunc.nout
    args = [lowerer.loadvar(name) for name in expr_args]
    return npyimpl.numpy_ufunc_kernel(context, builder, outer_sig, args, ufunc, ExprKernel)