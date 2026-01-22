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
class Fixedpoint(Z3PPObject):
    """Fixedpoint API provides methods for solving with recursive predicates"""

    def __init__(self, fixedpoint=None, ctx=None):
        assert fixedpoint is None or ctx is not None
        self.ctx = _get_ctx(ctx)
        self.fixedpoint = None
        if fixedpoint is None:
            self.fixedpoint = Z3_mk_fixedpoint(self.ctx.ref())
        else:
            self.fixedpoint = fixedpoint
        Z3_fixedpoint_inc_ref(self.ctx.ref(), self.fixedpoint)
        self.vars = []

    def __deepcopy__(self, memo={}):
        return FixedPoint(self.fixedpoint, self.ctx)

    def __del__(self):
        if self.fixedpoint is not None and self.ctx.ref() is not None and (Z3_fixedpoint_dec_ref is not None):
            Z3_fixedpoint_dec_ref(self.ctx.ref(), self.fixedpoint)

    def set(self, *args, **keys):
        """Set a configuration option. The method `help()` return a string containing all available options.
        """
        p = args2params(args, keys, self.ctx)
        Z3_fixedpoint_set_params(self.ctx.ref(), self.fixedpoint, p.params)

    def help(self):
        """Display a string describing all available options."""
        print(Z3_fixedpoint_get_help(self.ctx.ref(), self.fixedpoint))

    def param_descrs(self):
        """Return the parameter description set."""
        return ParamDescrsRef(Z3_fixedpoint_get_param_descrs(self.ctx.ref(), self.fixedpoint), self.ctx)

    def assert_exprs(self, *args):
        """Assert constraints as background axioms for the fixedpoint solver."""
        args = _get_args(args)
        s = BoolSort(self.ctx)
        for arg in args:
            if isinstance(arg, Goal) or isinstance(arg, AstVector):
                for f in arg:
                    f = self.abstract(f)
                    Z3_fixedpoint_assert(self.ctx.ref(), self.fixedpoint, f.as_ast())
            else:
                arg = s.cast(arg)
                arg = self.abstract(arg)
                Z3_fixedpoint_assert(self.ctx.ref(), self.fixedpoint, arg.as_ast())

    def add(self, *args):
        """Assert constraints as background axioms for the fixedpoint solver. Alias for assert_expr."""
        self.assert_exprs(*args)

    def __iadd__(self, fml):
        self.add(fml)
        return self

    def append(self, *args):
        """Assert constraints as background axioms for the fixedpoint solver. Alias for assert_expr."""
        self.assert_exprs(*args)

    def insert(self, *args):
        """Assert constraints as background axioms for the fixedpoint solver. Alias for assert_expr."""
        self.assert_exprs(*args)

    def add_rule(self, head, body=None, name=None):
        """Assert rules defining recursive predicates to the fixedpoint solver.
        >>> a = Bool('a')
        >>> b = Bool('b')
        >>> s = Fixedpoint()
        >>> s.register_relation(a.decl())
        >>> s.register_relation(b.decl())
        >>> s.fact(a)
        >>> s.rule(b, a)
        >>> s.query(b)
        sat
        """
        if name is None:
            name = ''
        name = to_symbol(name, self.ctx)
        if body is None:
            head = self.abstract(head)
            Z3_fixedpoint_add_rule(self.ctx.ref(), self.fixedpoint, head.as_ast(), name)
        else:
            body = _get_args(body)
            f = self.abstract(Implies(And(body, self.ctx), head))
            Z3_fixedpoint_add_rule(self.ctx.ref(), self.fixedpoint, f.as_ast(), name)

    def rule(self, head, body=None, name=None):
        """Assert rules defining recursive predicates to the fixedpoint solver. Alias for add_rule."""
        self.add_rule(head, body, name)

    def fact(self, head, name=None):
        """Assert facts defining recursive predicates to the fixedpoint solver. Alias for add_rule."""
        self.add_rule(head, None, name)

    def query(self, *query):
        """Query the fixedpoint engine whether formula is derivable.
           You can also pass an tuple or list of recursive predicates.
        """
        query = _get_args(query)
        sz = len(query)
        if sz >= 1 and isinstance(query[0], FuncDeclRef):
            _decls = (FuncDecl * sz)()
            i = 0
            for q in query:
                _decls[i] = q.ast
                i = i + 1
            r = Z3_fixedpoint_query_relations(self.ctx.ref(), self.fixedpoint, sz, _decls)
        else:
            if sz == 1:
                query = query[0]
            else:
                query = And(query, self.ctx)
            query = self.abstract(query, False)
            r = Z3_fixedpoint_query(self.ctx.ref(), self.fixedpoint, query.as_ast())
        return CheckSatResult(r)

    def query_from_lvl(self, lvl, *query):
        """Query the fixedpoint engine whether formula is derivable starting at the given query level.
        """
        query = _get_args(query)
        sz = len(query)
        if sz >= 1 and isinstance(query[0], FuncDecl):
            _z3_assert(False, 'unsupported')
        else:
            if sz == 1:
                query = query[0]
            else:
                query = And(query)
            query = self.abstract(query, False)
            r = Z3_fixedpoint_query_from_lvl(self.ctx.ref(), self.fixedpoint, query.as_ast(), lvl)
        return CheckSatResult(r)

    def update_rule(self, head, body, name):
        """update rule"""
        if name is None:
            name = ''
        name = to_symbol(name, self.ctx)
        body = _get_args(body)
        f = self.abstract(Implies(And(body, self.ctx), head))
        Z3_fixedpoint_update_rule(self.ctx.ref(), self.fixedpoint, f.as_ast(), name)

    def get_answer(self):
        """Retrieve answer from last query call."""
        r = Z3_fixedpoint_get_answer(self.ctx.ref(), self.fixedpoint)
        return _to_expr_ref(r, self.ctx)

    def get_ground_sat_answer(self):
        """Retrieve a ground cex from last query call."""
        r = Z3_fixedpoint_get_ground_sat_answer(self.ctx.ref(), self.fixedpoint)
        return _to_expr_ref(r, self.ctx)

    def get_rules_along_trace(self):
        """retrieve rules along the counterexample trace"""
        return AstVector(Z3_fixedpoint_get_rules_along_trace(self.ctx.ref(), self.fixedpoint), self.ctx)

    def get_rule_names_along_trace(self):
        """retrieve rule names along the counterexample trace"""
        names = _symbol2py(self.ctx, Z3_fixedpoint_get_rule_names_along_trace(self.ctx.ref(), self.fixedpoint))
        return names.split(';')

    def get_num_levels(self, predicate):
        """Retrieve number of levels used for predicate in PDR engine"""
        return Z3_fixedpoint_get_num_levels(self.ctx.ref(), self.fixedpoint, predicate.ast)

    def get_cover_delta(self, level, predicate):
        """Retrieve properties known about predicate for the level'th unfolding.
        -1 is treated as the limit (infinity)
        """
        r = Z3_fixedpoint_get_cover_delta(self.ctx.ref(), self.fixedpoint, level, predicate.ast)
        return _to_expr_ref(r, self.ctx)

    def add_cover(self, level, predicate, property):
        """Add property to predicate for the level'th unfolding.
        -1 is treated as infinity (infinity)
        """
        Z3_fixedpoint_add_cover(self.ctx.ref(), self.fixedpoint, level, predicate.ast, property.ast)

    def register_relation(self, *relations):
        """Register relation as recursive"""
        relations = _get_args(relations)
        for f in relations:
            Z3_fixedpoint_register_relation(self.ctx.ref(), self.fixedpoint, f.ast)

    def set_predicate_representation(self, f, *representations):
        """Control how relation is represented"""
        representations = _get_args(representations)
        representations = [to_symbol(s) for s in representations]
        sz = len(representations)
        args = (Symbol * sz)()
        for i in range(sz):
            args[i] = representations[i]
        Z3_fixedpoint_set_predicate_representation(self.ctx.ref(), self.fixedpoint, f.ast, sz, args)

    def parse_string(self, s):
        """Parse rules and queries from a string"""
        return AstVector(Z3_fixedpoint_from_string(self.ctx.ref(), self.fixedpoint, s), self.ctx)

    def parse_file(self, f):
        """Parse rules and queries from a file"""
        return AstVector(Z3_fixedpoint_from_file(self.ctx.ref(), self.fixedpoint, f), self.ctx)

    def get_rules(self):
        """retrieve rules that have been added to fixedpoint context"""
        return AstVector(Z3_fixedpoint_get_rules(self.ctx.ref(), self.fixedpoint), self.ctx)

    def get_assertions(self):
        """retrieve assertions that have been added to fixedpoint context"""
        return AstVector(Z3_fixedpoint_get_assertions(self.ctx.ref(), self.fixedpoint), self.ctx)

    def __repr__(self):
        """Return a formatted string with all added rules and constraints."""
        return self.sexpr()

    def sexpr(self):
        """Return a formatted string (in Lisp-like format) with all added constraints.
        We say the string is in s-expression format.
        """
        return Z3_fixedpoint_to_string(self.ctx.ref(), self.fixedpoint, 0, (Ast * 0)())

    def to_string(self, queries):
        """Return a formatted string (in Lisp-like format) with all added constraints.
           We say the string is in s-expression format.
           Include also queries.
        """
        args, len = _to_ast_array(queries)
        return Z3_fixedpoint_to_string(self.ctx.ref(), self.fixedpoint, len, args)

    def statistics(self):
        """Return statistics for the last `query()`.
        """
        return Statistics(Z3_fixedpoint_get_statistics(self.ctx.ref(), self.fixedpoint), self.ctx)

    def reason_unknown(self):
        """Return a string describing why the last `query()` returned `unknown`.
        """
        return Z3_fixedpoint_get_reason_unknown(self.ctx.ref(), self.fixedpoint)

    def declare_var(self, *vars):
        """Add variable or several variables.
        The added variable or variables will be bound in the rules
        and queries
        """
        vars = _get_args(vars)
        for v in vars:
            self.vars += [v]

    def abstract(self, fml, is_forall=True):
        if self.vars == []:
            return fml
        if is_forall:
            return ForAll(self.vars, fml)
        else:
            return Exists(self.vars, fml)