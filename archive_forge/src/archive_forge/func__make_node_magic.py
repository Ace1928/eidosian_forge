import builtins
import itertools
import logging
import math
import operator
import sys
from functools import lru_cache
from typing import Optional, Type, TYPE_CHECKING, Union
from torch import (  # noqa: F401
from torch.fx.experimental._sym_dispatch_mode import (
def _make_node_magic(method, func):
    func = lru_cache(256)(func)
    if method in magic_methods_on_operator_with_trailing_underscore:
        method_attr = f'{method}_'
    else:
        method_attr = method

    def binary_magic_impl(self, other):
        from torch.fx.experimental.symbolic_shapes import safe_expand
        op = method_to_operator(method)
        out_hint = None
        if self.hint is not None and other.hint is not None:
            out_hint = op(self.hint, other.hint)
        alternate_impl = alternate_impl_if_hinted_methods.get(method)
        if alternate_impl and out_hint is not None:
            return to_node(self, alternate_impl(wrap_node(self), wrap_node(other)))
        if sym_function_mode():
            return to_node(self, handle_sym_dispatch(op, (wrap_node(self), wrap_node(other)), {}))
        assert isinstance(other, SymNode)
        try:
            out = func(self.expr, other.expr)
        except Exception:
            log.warning('failed to eval %s(%s, %s)', method, self.expr, other.expr)
            raise
        out = safe_expand(out)
        pytype: Type
        if method in always_float_magic_methods:
            pytype = float
        elif method in always_bool_magic_methods:
            pytype = bool
        elif self.pytype is float or other.pytype is float:
            pytype = float
        else:
            pytype = self.pytype
        if pytype is not None and out_hint is not None and (not isinstance(out_hint, SymTypes)):
            out_hint = pytype(out_hint)
        fx_node, _ = self.shape_env.create_fx_call_function(op, (self.fx_node, other.fx_node))
        return SymNode(out, self.shape_env, pytype, out_hint, fx_node=fx_node)

    def unary_magic_impl(self):
        from torch.fx.experimental.symbolic_shapes import safe_expand
        op = method_to_operator(method)
        if sym_function_mode():
            return to_node(self, handle_sym_dispatch(op, (wrap_node(self),), {}))
        expr = self.expr
        if method == 'floor' or method == 'ceiling':
            expr = self.shape_env._simplify_floor_div(expr)
        try:
            out = func(expr)
        except Exception:
            log.warning('failed to eval %s(%s)', method, expr)
            raise
        out_hint = None
        if self.hint is not None:
            out_hint = op(self.hint)
        out = safe_expand(out)
        pytype: Type
        if method in always_int_magic_methods:
            pytype = int
        elif method in always_float_magic_methods:
            pytype = float
        else:
            pytype = self.pytype
        fx_node, _ = self.shape_env.create_fx_call_function(op, (self.fx_node,))
        return SymNode(out, self.shape_env, pytype, out_hint, fx_node=fx_node)
    if method in unary_magic_methods:
        setattr(SymNode, f'_{method_attr}', unary_magic_impl)
    elif method == 'sym_ite':

        def sym_ite_impl(pred_node, then_node, else_node):
            from torch.fx.experimental.symbolic_shapes import safe_expand
            out_hint = then_node.hint if pred_node.hint else else_node.hint
            if sym_function_mode():
                return to_node(pred_node, handle_sym_dispatch(sym_ite, (wrap_node(pred_node), wrap_node(then_node), wrap_node(else_node)), {}))
            try:
                out = func(pred_node.expr, then_node.expr, else_node.expr)
            except Exception:
                log.warning('failed to eval %s(%s, %s, %s)', method, pred_node.expr, then_node.expr, else_node.expr)
                raise
            out = safe_expand(out)
            fx_node, _ = pred_node.shape_env.create_fx_call_function(sym_ite, (pred_node.fx_node, then_node.fx_node, else_node.fx_node))
            return SymNode(out, pred_node.shape_env, then_node.pytype, out_hint, fx_node=fx_node)
        setattr(SymNode, f'_{method_attr}', sym_ite_impl)
    else:
        setattr(SymNode, f'_{method_attr}', binary_magic_impl)