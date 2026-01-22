from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class ModelRef(Z3PPObject):
    """Model/Solution of a satisfiability problem (aka system of constraints)."""

    def __init__(self, m, ctx):
        assert ctx is not None
        self.model = m
        self.ctx = ctx
        Z3_model_inc_ref(self.ctx.ref(), self.model)

    def __del__(self):
        if self.ctx.ref() is not None and Z3_model_dec_ref is not None:
            Z3_model_dec_ref(self.ctx.ref(), self.model)

    def __repr__(self):
        return obj_to_string(self)

    def sexpr(self):
        """Return a textual representation of the s-expression representing the model."""
        return Z3_model_to_string(self.ctx.ref(), self.model)

    def eval(self, t, model_completion=False):
        """Evaluate the expression `t` in the model `self`.
        If `model_completion` is enabled, then a default interpretation is automatically added
        for symbols that do not have an interpretation in the model `self`.

        >>> x = Int('x')
        >>> s = Solver()
        >>> s.add(x > 0, x < 2)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m.eval(x + 1)
        2
        >>> m.eval(x == 1)
        True
        >>> y = Int('y')
        >>> m.eval(y + x)
        1 + y
        >>> m.eval(y)
        y
        >>> m.eval(y, model_completion=True)
        0
        >>> # Now, m contains an interpretation for y
        >>> m.eval(y + x)
        1
        """
        r = (Ast * 1)()
        if Z3_model_eval(self.ctx.ref(), self.model, t.as_ast(), model_completion, r):
            return _to_expr_ref(r[0], self.ctx)
        raise Z3Exception('failed to evaluate expression in the model')

    def evaluate(self, t, model_completion=False):
        """Alias for `eval`.

        >>> x = Int('x')
        >>> s = Solver()
        >>> s.add(x > 0, x < 2)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m.evaluate(x + 1)
        2
        >>> m.evaluate(x == 1)
        True
        >>> y = Int('y')
        >>> m.evaluate(y + x)
        1 + y
        >>> m.evaluate(y)
        y
        >>> m.evaluate(y, model_completion=True)
        0
        >>> # Now, m contains an interpretation for y
        >>> m.evaluate(y + x)
        1
        """
        return self.eval(t, model_completion)

    def __len__(self):
        """Return the number of constant and function declarations in the model `self`.

        >>> f = Function('f', IntSort(), IntSort())
        >>> x = Int('x')
        >>> s = Solver()
        >>> s.add(x > 0, f(x) != x)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> len(m)
        2
        """
        num_consts = int(Z3_model_get_num_consts(self.ctx.ref(), self.model))
        num_funcs = int(Z3_model_get_num_funcs(self.ctx.ref(), self.model))
        return num_consts + num_funcs

    def get_interp(self, decl):
        """Return the interpretation for a given declaration or constant.

        >>> f = Function('f', IntSort(), IntSort())
        >>> x = Int('x')
        >>> s = Solver()
        >>> s.add(x > 0, x < 2, f(x) == 0)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m[x]
        1
        >>> m[f]
        [else -> 0]
        """
        if z3_debug():
            _z3_assert(isinstance(decl, FuncDeclRef) or is_const(decl), 'Z3 declaration expected')
        if is_const(decl):
            decl = decl.decl()
        try:
            if decl.arity() == 0:
                _r = Z3_model_get_const_interp(self.ctx.ref(), self.model, decl.ast)
                if _r.value is None:
                    return None
                r = _to_expr_ref(_r, self.ctx)
                if is_as_array(r):
                    fi = self.get_interp(get_as_array_func(r))
                    if fi is None:
                        return fi
                    e = fi.else_value()
                    if e is None:
                        return fi
                    if fi.arity() != 1:
                        return fi
                    srt = decl.range()
                    dom = srt.domain()
                    e = K(dom, e)
                    i = 0
                    sz = fi.num_entries()
                    n = fi.arity()
                    while i < sz:
                        fe = fi.entry(i)
                        e = Store(e, fe.arg_value(0), fe.value())
                        i += 1
                    return e
                else:
                    return r
            else:
                return FuncInterp(Z3_model_get_func_interp(self.ctx.ref(), self.model, decl.ast), self.ctx)
        except Z3Exception:
            return None

    def num_sorts(self):
        """Return the number of uninterpreted sorts that contain an interpretation in the model `self`.

        >>> A = DeclareSort('A')
        >>> a, b = Consts('a b', A)
        >>> s = Solver()
        >>> s.add(a != b)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m.num_sorts()
        1
        """
        return int(Z3_model_get_num_sorts(self.ctx.ref(), self.model))

    def get_sort(self, idx):
        """Return the uninterpreted sort at position `idx` < self.num_sorts().

        >>> A = DeclareSort('A')
        >>> B = DeclareSort('B')
        >>> a1, a2 = Consts('a1 a2', A)
        >>> b1, b2 = Consts('b1 b2', B)
        >>> s = Solver()
        >>> s.add(a1 != a2, b1 != b2)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m.num_sorts()
        2
        >>> m.get_sort(0)
        A
        >>> m.get_sort(1)
        B
        """
        if idx >= self.num_sorts():
            raise IndexError
        return _to_sort_ref(Z3_model_get_sort(self.ctx.ref(), self.model, idx), self.ctx)

    def sorts(self):
        """Return all uninterpreted sorts that have an interpretation in the model `self`.

        >>> A = DeclareSort('A')
        >>> B = DeclareSort('B')
        >>> a1, a2 = Consts('a1 a2', A)
        >>> b1, b2 = Consts('b1 b2', B)
        >>> s = Solver()
        >>> s.add(a1 != a2, b1 != b2)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m.sorts()
        [A, B]
        """
        return [self.get_sort(i) for i in range(self.num_sorts())]

    def get_universe(self, s):
        """Return the interpretation for the uninterpreted sort `s` in the model `self`.

        >>> A = DeclareSort('A')
        >>> a, b = Consts('a b', A)
        >>> s = Solver()
        >>> s.add(a != b)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m.get_universe(A)
        [A!val!1, A!val!0]
        """
        if z3_debug():
            _z3_assert(isinstance(s, SortRef), 'Z3 sort expected')
        try:
            return AstVector(Z3_model_get_sort_universe(self.ctx.ref(), self.model, s.ast), self.ctx)
        except Z3Exception:
            return None

    def __getitem__(self, idx):
        """If `idx` is an integer, then the declaration at position `idx` in the model `self` is returned.
        If `idx` is a declaration, then the actual interpretation is returned.

        The elements can be retrieved using position or the actual declaration.

        >>> f = Function('f', IntSort(), IntSort())
        >>> x = Int('x')
        >>> s = Solver()
        >>> s.add(x > 0, x < 2, f(x) == 0)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> len(m)
        2
        >>> m[0]
        x
        >>> m[1]
        f
        >>> m[x]
        1
        >>> m[f]
        [else -> 0]
        >>> for d in m: print("%s -> %s" % (d, m[d]))
        x -> 1
        f -> [else -> 0]
        """
        if _is_int(idx):
            if idx >= len(self):
                raise IndexError
            num_consts = Z3_model_get_num_consts(self.ctx.ref(), self.model)
            if idx < num_consts:
                return FuncDeclRef(Z3_model_get_const_decl(self.ctx.ref(), self.model, idx), self.ctx)
            else:
                return FuncDeclRef(Z3_model_get_func_decl(self.ctx.ref(), self.model, idx - num_consts), self.ctx)
        if isinstance(idx, FuncDeclRef):
            return self.get_interp(idx)
        if is_const(idx):
            return self.get_interp(idx.decl())
        if isinstance(idx, SortRef):
            return self.get_universe(idx)
        if z3_debug():
            _z3_assert(False, 'Integer, Z3 declaration, or Z3 constant expected')
        return None

    def decls(self):
        """Return a list with all symbols that have an interpretation in the model `self`.
        >>> f = Function('f', IntSort(), IntSort())
        >>> x = Int('x')
        >>> s = Solver()
        >>> s.add(x > 0, x < 2, f(x) == 0)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m.decls()
        [x, f]
        """
        r = []
        for i in range(Z3_model_get_num_consts(self.ctx.ref(), self.model)):
            r.append(FuncDeclRef(Z3_model_get_const_decl(self.ctx.ref(), self.model, i), self.ctx))
        for i in range(Z3_model_get_num_funcs(self.ctx.ref(), self.model)):
            r.append(FuncDeclRef(Z3_model_get_func_decl(self.ctx.ref(), self.model, i), self.ctx))
        return r

    def update_value(self, x, value):
        """Update the interpretation of a constant"""
        if is_expr(x):
            x = x.decl()
        if is_func_decl(x) and x.arity() != 0 and isinstance(value, FuncInterp):
            fi1 = value.f
            fi2 = Z3_add_func_interp(x.ctx_ref(), self.model, x.ast, value.else_value().ast)
            fi2 = FuncInterp(fi2, x.ctx)
            for i in range(value.num_entries()):
                e = value.entry(i)
                n = Z3_func_entry_get_num_args(x.ctx_ref(), e.entry)
                v = AstVector()
                for j in range(n):
                    v.push(e.arg_value(j))
                val = Z3_func_entry_get_value(x.ctx_ref(), e.entry)
                Z3_func_interp_add_entry(x.ctx_ref(), fi2.f, v.vector, val)
            return
        if not is_func_decl(x) or x.arity() != 0:
            raise Z3Exception('Expecting 0-ary function or constant expression')
        value = _py2expr(value)
        Z3_add_const_interp(x.ctx_ref(), self.model, x.ast, value.ast)

    def translate(self, target):
        """Translate `self` to the context `target`. That is, return a copy of `self` in the context `target`.
        """
        if z3_debug():
            _z3_assert(isinstance(target, Context), 'argument must be a Z3 context')
        model = Z3_model_translate(self.ctx.ref(), self.model, target.ref())
        return ModelRef(model, target)

    def __copy__(self):
        return self.translate(self.ctx)

    def __deepcopy__(self, memo={}):
        return self.translate(self.ctx)