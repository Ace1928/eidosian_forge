from sympy.core import S, Function, diff, Tuple, Dummy, Mul
from sympy.core.basic import Basic, as_Basic
from sympy.core.numbers import Rational, NumberSymbol, _illegal
from sympy.core.parameters import global_parameters
from sympy.core.relational import (Lt, Gt, Eq, Ne, Relational,
from sympy.core.sorting import ordered
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
from sympy.utilities.iterables import uniq, sift, common_prefix
from sympy.utilities.misc import filldedent, func_name
from itertools import product
class Piecewise(Function):
    """
    Represents a piecewise function.

    Usage:

      Piecewise( (expr,cond), (expr,cond), ... )
        - Each argument is a 2-tuple defining an expression and condition
        - The conds are evaluated in turn returning the first that is True.
          If any of the evaluated conds are not explicitly False,
          e.g. ``x < 1``, the function is returned in symbolic form.
        - If the function is evaluated at a place where all conditions are False,
          nan will be returned.
        - Pairs where the cond is explicitly False, will be removed and no pair
          appearing after a True condition will ever be retained. If a single
          pair with a True condition remains, it will be returned, even when
          evaluation is False.

    Examples
    ========

    >>> from sympy import Piecewise, log, piecewise_fold
    >>> from sympy.abc import x, y
    >>> f = x**2
    >>> g = log(x)
    >>> p = Piecewise((0, x < -1), (f, x <= 1), (g, True))
    >>> p.subs(x,1)
    1
    >>> p.subs(x,5)
    log(5)

    Booleans can contain Piecewise elements:

    >>> cond = (x < y).subs(x, Piecewise((2, x < 0), (3, True))); cond
    Piecewise((2, x < 0), (3, True)) < y

    The folded version of this results in a Piecewise whose
    expressions are Booleans:

    >>> folded_cond = piecewise_fold(cond); folded_cond
    Piecewise((2 < y, x < 0), (3 < y, True))

    When a Boolean containing Piecewise (like cond) or a Piecewise
    with Boolean expressions (like folded_cond) is used as a condition,
    it is converted to an equivalent :class:`~.ITE` object:

    >>> Piecewise((1, folded_cond))
    Piecewise((1, ITE(x < 0, y > 2, y > 3)))

    When a condition is an ``ITE``, it will be converted to a simplified
    Boolean expression:

    >>> piecewise_fold(_)
    Piecewise((1, ((x >= 0) | (y > 2)) & ((y > 3) | (x < 0))))

    See Also
    ========

    piecewise_fold
    piecewise_exclusive
    ITE
    """
    nargs = None
    is_Piecewise = True

    def __new__(cls, *args, **options):
        if len(args) == 0:
            raise TypeError('At least one (expr, cond) pair expected.')
        newargs = []
        for ec in args:
            pair = ExprCondPair(*getattr(ec, 'args', ec))
            cond = pair.cond
            if cond is false:
                continue
            newargs.append(pair)
            if cond is true:
                break
        eval = options.pop('evaluate', global_parameters.evaluate)
        if eval:
            r = cls.eval(*newargs)
            if r is not None:
                return r
        elif len(newargs) == 1 and newargs[0].cond == True:
            return newargs[0].expr
        return Basic.__new__(cls, *newargs, **options)

    @classmethod
    def eval(cls, *_args):
        """Either return a modified version of the args or, if no
        modifications were made, return None.

        Modifications that are made here:

        1. relationals are made canonical
        2. any False conditions are dropped
        3. any repeat of a previous condition is ignored
        4. any args past one with a true condition are dropped

        If there are no args left, nan will be returned.
        If there is a single arg with a True condition, its
        corresponding expression will be returned.

        EXAMPLES
        ========

        >>> from sympy import Piecewise
        >>> from sympy.abc import x
        >>> cond = -x < -1
        >>> args = [(1, cond), (4, cond), (3, False), (2, True), (5, x < 1)]
        >>> Piecewise(*args, evaluate=False)
        Piecewise((1, -x < -1), (4, -x < -1), (2, True))
        >>> Piecewise(*args)
        Piecewise((1, x > 1), (2, True))
        """
        if not _args:
            return Undefined
        if len(_args) == 1 and _args[0][-1] == True:
            return _args[0][0]
        newargs = _piecewise_collapse_arguments(_args)
        missing = len(newargs) != len(_args)
        same = all((a == b for a, b in zip(newargs, _args)))
        if not newargs:
            raise ValueError(filldedent('\n                There are no conditions (or none that\n                are not trivially false) to define an\n                expression.'))
        if missing or not same:
            return cls(*newargs)

    def doit(self, **hints):
        """
        Evaluate this piecewise function.
        """
        newargs = []
        for e, c in self.args:
            if hints.get('deep', True):
                if isinstance(e, Basic):
                    newe = e.doit(**hints)
                    if newe != self:
                        e = newe
                if isinstance(c, Basic):
                    c = c.doit(**hints)
            newargs.append((e, c))
        return self.func(*newargs)

    def _eval_simplify(self, **kwargs):
        return piecewise_simplify(self, **kwargs)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        for e, c in self.args:
            if c == True or c.subs(x, 0) == True:
                return e.as_leading_term(x)

    def _eval_adjoint(self):
        return self.func(*[(e.adjoint(), c) for e, c in self.args])

    def _eval_conjugate(self):
        return self.func(*[(e.conjugate(), c) for e, c in self.args])

    def _eval_derivative(self, x):
        return self.func(*[(diff(e, x), c) for e, c in self.args])

    def _eval_evalf(self, prec):
        return self.func(*[(e._evalf(prec), c) for e, c in self.args])

    def _eval_is_meromorphic(self, x, a):
        if not a.is_real:
            return None
        for e, c in self.args:
            cond = c.subs(x, a)
            if cond.is_Relational:
                return None
            if a in c.as_set().boundary:
                return None
            if cond:
                return e._eval_is_meromorphic(x, a)

    def piecewise_integrate(self, x, **kwargs):
        """Return the Piecewise with each expression being
        replaced with its antiderivative. To obtain a continuous
        antiderivative, use the :func:`~.integrate` function or method.

        Examples
        ========

        >>> from sympy import Piecewise
        >>> from sympy.abc import x
        >>> p = Piecewise((0, x < 0), (1, x < 1), (2, True))
        >>> p.piecewise_integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x, True))

        Note that this does not give a continuous function, e.g.
        at x = 1 the 3rd condition applies and the antiderivative
        there is 2*x so the value of the antiderivative is 2:

        >>> anti = _
        >>> anti.subs(x, 1)
        2

        The continuous derivative accounts for the integral *up to*
        the point of interest, however:

        >>> p.integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x - 1, True))
        >>> _.subs(x, 1)
        1

        See Also
        ========
        Piecewise._eval_integral
        """
        from sympy.integrals import integrate
        return self.func(*[(integrate(e, x, **kwargs), c) for e, c in self.args])

    def _handle_irel(self, x, handler):
        """Return either None (if the conditions of self depend only on x) else
        a Piecewise expression whose expressions (handled by the handler that
        was passed) are paired with the governing x-independent relationals,
        e.g. Piecewise((A, a(x) & b(y)), (B, c(x) | c(y)) ->
        Piecewise(
            (handler(Piecewise((A, a(x) & True), (B, c(x) | True)), b(y) & c(y)),
            (handler(Piecewise((A, a(x) & True), (B, c(x) | False)), b(y)),
            (handler(Piecewise((A, a(x) & False), (B, c(x) | True)), c(y)),
            (handler(Piecewise((A, a(x) & False), (B, c(x) | False)), True))
        """
        rel = self.atoms(Relational)
        irel = list(ordered([r for r in rel if x not in r.free_symbols and r not in (S.true, S.false)]))
        if irel:
            args = {}
            exprinorder = []
            for truth in product((1, 0), repeat=len(irel)):
                reps = dict(zip(irel, truth))
                if 1 not in truth:
                    cond = None
                else:
                    andargs = Tuple(*[i for i in reps if reps[i]])
                    free = list(andargs.free_symbols)
                    if len(free) == 1:
                        from sympy.solvers.inequalities import reduce_inequalities, _solve_inequality
                        try:
                            t = reduce_inequalities(andargs, free[0])
                        except (ValueError, NotImplementedError):
                            t = And(*[_solve_inequality(a, free[0], linear=True) for a in andargs])
                    else:
                        t = And(*andargs)
                    if t is S.false:
                        continue
                    cond = t
                expr = handler(self.xreplace(reps))
                if isinstance(expr, self.func) and len(expr.args) == 1:
                    expr, econd = expr.args[0]
                    cond = And(econd, True if cond is None else cond)
                if cond is not None:
                    args.setdefault(expr, []).append(cond)
                    exprinorder.append(expr)
            for k in args:
                args[k] = Or(*args[k])
            args = [(e, args[e]) for e in uniq(exprinorder)]
            args.append((expr, True))
            return Piecewise(*args)

    def _eval_integral(self, x, _first=True, **kwargs):
        """Return the indefinite integral of the
        Piecewise such that subsequent substitution of x with a
        value will give the value of the integral (not including
        the constant of integration) up to that point. To only
        integrate the individual parts of Piecewise, use the
        ``piecewise_integrate`` method.

        Examples
        ========

        >>> from sympy import Piecewise
        >>> from sympy.abc import x
        >>> p = Piecewise((0, x < 0), (1, x < 1), (2, True))
        >>> p.integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x - 1, True))
        >>> p.piecewise_integrate(x)
        Piecewise((0, x < 0), (x, x < 1), (2*x, True))

        See Also
        ========
        Piecewise.piecewise_integrate
        """
        from sympy.integrals.integrals import integrate
        if _first:

            def handler(ipw):
                if isinstance(ipw, self.func):
                    return ipw._eval_integral(x, _first=False, **kwargs)
                else:
                    return ipw.integrate(x, **kwargs)
            irv = self._handle_irel(x, handler)
            if irv is not None:
                return irv
        ok, abei = self._intervals(x)
        if not ok:
            from sympy.integrals.integrals import Integral
            return Integral(self, x)
        pieces = [(a, b) for a, b, _, _ in abei]
        oo = S.Infinity
        done = [(-oo, oo, -1)]
        for k, p in enumerate(pieces):
            if p == (-oo, oo):
                for j, (a, b, i) in enumerate(done):
                    if i == -1:
                        done[j] = (a, b, k)
                break
            N = len(done) - 1
            for j, (a, b, i) in enumerate(reversed(done)):
                if i == -1:
                    j = N - j
                    done[j:j + 1] = _clip(p, (a, b), k)
        done = [(a, b, i) for a, b, i in done if a != b]
        if any((i == -1 for a, b, i in done)):
            abei.append((-oo, oo, Undefined, -1))
        args = []
        sum = None
        for a, b, i in done:
            anti = integrate(abei[i][-2], x, **kwargs)
            if sum is None:
                sum = anti
            else:
                sum = sum.subs(x, a)
                e = anti._eval_interval(x, a, x)
                if sum.has(*_illegal) or e.has(*_illegal):
                    sum = anti
                else:
                    sum += e
            if b is S.Infinity:
                cond = True
            elif self.args[abei[i][-1]].cond.subs(x, b) == False:
                cond = x < b
            else:
                cond = x <= b
            args.append((sum, cond))
        return Piecewise(*args)

    def _eval_interval(self, sym, a, b, _first=True):
        """Evaluates the function along the sym in a given interval [a, b]"""
        if a is None or b is None:
            return super()._eval_interval(sym, a, b)
        else:
            x, lo, hi = map(as_Basic, (sym, a, b))
        if _first:

            def handler(ipw):
                if isinstance(ipw, self.func):
                    return ipw._eval_interval(x, lo, hi, _first=None)
                else:
                    return ipw._eval_interval(x, lo, hi)
            irv = self._handle_irel(x, handler)
            if irv is not None:
                return irv
            if (lo < hi) is S.false or (lo is S.Infinity or hi is S.NegativeInfinity):
                rv = self._eval_interval(x, hi, lo, _first=False)
                if isinstance(rv, Piecewise):
                    rv = Piecewise(*[(-e, c) for e, c in rv.args])
                else:
                    rv = -rv
                return rv
            if (lo < hi) is S.true or (hi is S.Infinity or lo is S.NegativeInfinity):
                pass
            else:
                _a = Dummy('lo')
                _b = Dummy('hi')
                a = lo if lo.is_comparable else _a
                b = hi if hi.is_comparable else _b
                pos = self._eval_interval(x, a, b, _first=False)
                if a == _a and b == _b:
                    neg, pos = (-pos.xreplace({_a: hi, _b: lo}), pos.xreplace({_a: lo, _b: hi}))
                else:
                    neg, pos = (-self._eval_interval(x, hi, lo, _first=False), pos.xreplace({_a: lo, _b: hi}))
                p = Dummy('', positive=True)
                if lo.is_Symbol:
                    pos = pos.xreplace({lo: hi - p}).xreplace({p: hi - lo})
                    neg = neg.xreplace({lo: hi + p}).xreplace({p: lo - hi})
                elif hi.is_Symbol:
                    pos = pos.xreplace({hi: lo + p}).xreplace({p: hi - lo})
                    neg = neg.xreplace({hi: lo - p}).xreplace({p: lo - hi})
                touch = lambda _: _.replace(lambda x: isinstance(x, (Min, Max)), lambda x: x.func(*x.args))
                neg = touch(neg)
                pos = touch(pos)
                if a == _a:
                    rv = Piecewise((pos, lo < hi), (neg, True))
                else:
                    rv = Piecewise((neg, hi < lo), (pos, True))
                if rv == Undefined:
                    raise ValueError("Can't integrate across undefined region.")
                if any((isinstance(i, Piecewise) for i in (pos, neg))):
                    rv = piecewise_fold(rv)
                return rv
        ok, abei = self._intervals(x)
        if not ok:
            from sympy.integrals.integrals import Integral
            return Integral(self.diff(x), (x, lo, hi))
        pieces = [(a, b) for a, b, _, _ in abei]
        done = [(lo, hi, -1)]
        oo = S.Infinity
        for k, p in enumerate(pieces):
            if p[:2] == (-oo, oo):
                for j, (a, b, i) in enumerate(done):
                    if i == -1:
                        done[j] = (a, b, k)
                break
            N = len(done) - 1
            for j, (a, b, i) in enumerate(reversed(done)):
                if i == -1:
                    j = N - j
                    done[j:j + 1] = _clip(p, (a, b), k)
        done = [(a, b, i) for a, b, i in done if a != b]
        sum = S.Zero
        upto = None
        for a, b, i in done:
            if i == -1:
                if upto is None:
                    return Undefined
                return Piecewise((sum, hi <= upto), (Undefined, True))
            sum += abei[i][-2]._eval_interval(x, a, b)
            upto = b
        return sum

    def _intervals(self, sym, err_on_Eq=False):
        """Return a bool and a message (when bool is False), else a
        list of unique tuples, (a, b, e, i), where a and b
        are the lower and upper bounds in which the expression e of
        argument i in self is defined and $a < b$ (when involving
        numbers) or $a \\le b$ when involving symbols.

        If there are any relationals not involving sym, or any
        relational cannot be solved for sym, the bool will be False
        a message be given as the second return value. The calling
        routine should have removed such relationals before calling
        this routine.

        The evaluated conditions will be returned as ranges.
        Discontinuous ranges will be returned separately with
        identical expressions. The first condition that evaluates to
        True will be returned as the last tuple with a, b = -oo, oo.
        """
        from sympy.solvers.inequalities import _solve_inequality
        assert isinstance(self, Piecewise)

        def nonsymfail(cond):
            return (False, filldedent('\n                A condition not involving\n                %s appeared: %s' % (sym, cond)))

        def _solve_relational(r):
            if sym not in r.free_symbols:
                return nonsymfail(r)
            try:
                rv = _solve_inequality(r, sym)
            except NotImplementedError:
                return (False, 'Unable to solve relational %s for %s.' % (r, sym))
            if isinstance(rv, Relational):
                free = rv.args[1].free_symbols
                if rv.args[0] != sym or sym in free:
                    return (False, 'Unable to solve relational %s for %s.' % (r, sym))
                if rv.rel_op == '==':
                    rv = S.false
                elif rv.rel_op == '!=':
                    try:
                        rv = Or(sym < rv.rhs, sym > rv.rhs)
                    except TypeError:
                        rv = S.true
            elif rv == (S.NegativeInfinity < sym) & (sym < S.Infinity):
                rv = S.true
            return (True, rv)
        args = list(self.args)
        keys = self.atoms(Relational)
        reps = {}
        for r in keys:
            ok, s = _solve_relational(r)
            if ok != True:
                return (False, ok)
            reps[r] = s
        args = [i.xreplace(reps) for i in self.args]
        expr_cond = []
        default = idefault = None
        for i, (expr, cond) in enumerate(args):
            if cond is S.false:
                continue
            if cond is S.true:
                default = expr
                idefault = i
                break
            if isinstance(cond, Eq):
                if err_on_Eq:
                    return (False, 'encountered Eq condition: %s' % cond)
                continue
            cond = to_cnf(cond)
            if isinstance(cond, And):
                cond = distribute_or_over_and(cond)
            if isinstance(cond, Or):
                expr_cond.extend([(i, expr, o) for o in cond.args if not isinstance(o, Eq)])
            elif cond is not S.false:
                expr_cond.append((i, expr, cond))
            elif cond is S.true:
                default = expr
                idefault = i
                break
        int_expr = []
        for iarg, expr, cond in expr_cond:
            if isinstance(cond, And):
                lower = S.NegativeInfinity
                upper = S.Infinity
                exclude = []
                for cond2 in cond.args:
                    if not isinstance(cond2, Relational):
                        return (False, 'expecting only Relationals')
                    if isinstance(cond2, Eq):
                        lower = upper
                        if err_on_Eq:
                            return (False, 'encountered secondary Eq condition')
                        break
                    elif isinstance(cond2, Ne):
                        l, r = cond2.args
                        if l == sym:
                            exclude.append(r)
                        elif r == sym:
                            exclude.append(l)
                        else:
                            return nonsymfail(cond2)
                        continue
                    elif cond2.lts == sym:
                        upper = Min(cond2.gts, upper)
                    elif cond2.gts == sym:
                        lower = Max(cond2.lts, lower)
                    else:
                        return nonsymfail(cond2)
                if exclude:
                    exclude = list(ordered(exclude))
                    newcond = []
                    for i, e in enumerate(exclude):
                        if e < lower == True or e > upper == True:
                            continue
                        if not newcond:
                            newcond.append((None, lower))
                        newcond.append((newcond[-1][1], e))
                    newcond.append((newcond[-1][1], upper))
                    newcond.pop(0)
                    expr_cond.extend([(iarg, expr, And(i[0] < sym, sym < i[1])) for i in newcond])
                    continue
            elif isinstance(cond, Relational) and cond.rel_op != '!=':
                lower, upper = (cond.lts, cond.gts)
                if cond.lts == sym:
                    lower = S.NegativeInfinity
                elif cond.gts == sym:
                    upper = S.Infinity
                else:
                    return nonsymfail(cond)
            else:
                return (False, 'unrecognized condition: %s' % cond)
            lower, upper = (lower, Max(lower, upper))
            if err_on_Eq and lower == upper:
                return (False, 'encountered Eq condition')
            if (lower >= upper) is not S.true:
                int_expr.append((lower, upper, expr, iarg))
        if default is not None:
            int_expr.append((S.NegativeInfinity, S.Infinity, default, idefault))
        return (True, list(uniq(int_expr)))

    def _eval_nseries(self, x, n, logx, cdir=0):
        args = [(ec.expr._eval_nseries(x, n, logx), ec.cond) for ec in self.args]
        return self.func(*args)

    def _eval_power(self, s):
        return self.func(*[(e ** s, c) for e, c in self.args])

    def _eval_subs(self, old, new):
        args = list(self.args)
        args_exist = False
        for i, (e, c) in enumerate(args):
            c = c._subs(old, new)
            if c != False:
                args_exist = True
                e = e._subs(old, new)
            args[i] = (e, c)
            if c == True:
                break
        if not args_exist:
            args = ((Undefined, True),)
        return self.func(*args)

    def _eval_transpose(self):
        return self.func(*[(e.transpose(), c) for e, c in self.args])

    def _eval_template_is_attr(self, is_attr):
        b = None
        for expr, _ in self.args:
            a = getattr(expr, is_attr)
            if a is None:
                return
            if b is None:
                b = a
            elif b is not a:
                return
        return b
    _eval_is_finite = lambda self: self._eval_template_is_attr('is_finite')
    _eval_is_complex = lambda self: self._eval_template_is_attr('is_complex')
    _eval_is_even = lambda self: self._eval_template_is_attr('is_even')
    _eval_is_imaginary = lambda self: self._eval_template_is_attr('is_imaginary')
    _eval_is_integer = lambda self: self._eval_template_is_attr('is_integer')
    _eval_is_irrational = lambda self: self._eval_template_is_attr('is_irrational')
    _eval_is_negative = lambda self: self._eval_template_is_attr('is_negative')
    _eval_is_nonnegative = lambda self: self._eval_template_is_attr('is_nonnegative')
    _eval_is_nonpositive = lambda self: self._eval_template_is_attr('is_nonpositive')
    _eval_is_nonzero = lambda self: self._eval_template_is_attr('is_nonzero')
    _eval_is_odd = lambda self: self._eval_template_is_attr('is_odd')
    _eval_is_polar = lambda self: self._eval_template_is_attr('is_polar')
    _eval_is_positive = lambda self: self._eval_template_is_attr('is_positive')
    _eval_is_extended_real = lambda self: self._eval_template_is_attr('is_extended_real')
    _eval_is_extended_positive = lambda self: self._eval_template_is_attr('is_extended_positive')
    _eval_is_extended_negative = lambda self: self._eval_template_is_attr('is_extended_negative')
    _eval_is_extended_nonzero = lambda self: self._eval_template_is_attr('is_extended_nonzero')
    _eval_is_extended_nonpositive = lambda self: self._eval_template_is_attr('is_extended_nonpositive')
    _eval_is_extended_nonnegative = lambda self: self._eval_template_is_attr('is_extended_nonnegative')
    _eval_is_real = lambda self: self._eval_template_is_attr('is_real')
    _eval_is_zero = lambda self: self._eval_template_is_attr('is_zero')

    @classmethod
    def __eval_cond(cls, cond):
        """Return the truth value of the condition."""
        if cond == True:
            return True
        if isinstance(cond, Eq):
            try:
                diff = cond.lhs - cond.rhs
                if diff.is_commutative:
                    return diff.is_zero
            except TypeError:
                pass

    def as_expr_set_pairs(self, domain=None):
        """Return tuples for each argument of self that give
        the expression and the interval in which it is valid
        which is contained within the given domain.
        If a condition cannot be converted to a set, an error
        will be raised. The variable of the conditions is
        assumed to be real; sets of real values are returned.

        Examples
        ========

        >>> from sympy import Piecewise, Interval
        >>> from sympy.abc import x
        >>> p = Piecewise(
        ...     (1, x < 2),
        ...     (2,(x > 0) & (x < 4)),
        ...     (3, True))
        >>> p.as_expr_set_pairs()
        [(1, Interval.open(-oo, 2)),
         (2, Interval.Ropen(2, 4)),
         (3, Interval(4, oo))]
        >>> p.as_expr_set_pairs(Interval(0, 3))
        [(1, Interval.Ropen(0, 2)),
         (2, Interval(2, 3))]
        """
        if domain is None:
            domain = S.Reals
        exp_sets = []
        U = domain
        complex = not domain.is_subset(S.Reals)
        cond_free = set()
        for expr, cond in self.args:
            cond_free |= cond.free_symbols
            if len(cond_free) > 1:
                raise NotImplementedError(filldedent('\n                    multivariate conditions are not handled.'))
            if complex:
                for i in cond.atoms(Relational):
                    if not isinstance(i, (Eq, Ne)):
                        raise ValueError(filldedent('\n                            Inequalities in the complex domain are\n                            not supported. Try the real domain by\n                            setting domain=S.Reals'))
            cond_int = U.intersect(cond.as_set())
            U = U - cond_int
            if cond_int != S.EmptySet:
                exp_sets.append((expr, cond_int))
        return exp_sets

    def _eval_rewrite_as_ITE(self, *args, **kwargs):
        byfree = {}
        args = list(args)
        default = any((c == True for b, c in args))
        for i, (b, c) in enumerate(args):
            if not isinstance(b, Boolean) and b != True:
                raise TypeError(filldedent('\n                    Expecting Boolean or bool but got `%s`\n                    ' % func_name(b)))
            if c == True:
                break
            for c in c.args if isinstance(c, Or) else [c]:
                free = c.free_symbols
                x = free.pop()
                try:
                    byfree[x] = byfree.setdefault(x, S.EmptySet).union(c.as_set())
                except NotImplementedError:
                    if not default:
                        raise NotImplementedError(filldedent('\n                            A method to determine whether a multivariate\n                            conditional is consistent with a complete coverage\n                            of all variables has not been implemented so the\n                            rewrite is being stopped after encountering `%s`.\n                            This error would not occur if a default expression\n                            like `(foo, True)` were given.\n                            ' % c))
                if byfree[x] in (S.UniversalSet, S.Reals):
                    args[i] = list(args[i])
                    c = args[i][1] = True
                    break
            if c == True:
                break
        if c != True:
            raise ValueError(filldedent('\n                Conditions must cover all reals or a final default\n                condition `(foo, True)` must be given.\n                '))
        last, _ = args[i]
        for a, c in reversed(args[:i]):
            last = ITE(c, a, last)
        return _canonical(last)

    def _eval_rewrite_as_KroneckerDelta(self, *args):
        from sympy.functions.special.tensor_functions import KroneckerDelta
        rules = {And: [False, False], Or: [True, True], Not: [True, False], Eq: [None, None], Ne: [None, None]}

        class UnrecognizedCondition(Exception):
            pass

        def rewrite(cond):
            if isinstance(cond, Eq):
                return KroneckerDelta(*cond.args)
            if isinstance(cond, Ne):
                return 1 - KroneckerDelta(*cond.args)
            cls, args = (type(cond), cond.args)
            if cls not in rules:
                raise UnrecognizedCondition(cls)
            b1, b2 = rules[cls]
            k = Mul(*[1 - rewrite(c) for c in args]) if b1 else Mul(*[rewrite(c) for c in args])
            if b2:
                return 1 - k
            return k
        conditions = []
        true_value = None
        for value, cond in args:
            if type(cond) in rules:
                conditions.append((value, cond))
            elif cond is S.true:
                if true_value is None:
                    true_value = value
            else:
                return
        if true_value is not None:
            result = true_value
            for value, cond in conditions[::-1]:
                try:
                    k = rewrite(cond)
                    result = k * value + (1 - k) * result
                except UnrecognizedCondition:
                    return
            return result