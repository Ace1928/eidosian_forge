from itertools import product
from typing import Tuple as tTuple
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import (Function, ArgumentIndexError, expand_log,
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational, pi, I, ImaginaryUnit
from sympy.core.parameters import global_parameters
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import arg, unpolarify, im, re, Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory import multiplicity, perfect_power
from sympy.ntheory.factor_ import factorint
class log(Function):
    """
    The natural logarithm function `\\ln(x)` or `\\log(x)`.

    Explanation
    ===========

    Logarithms are taken with the natural base, `e`. To get
    a logarithm of a different base ``b``, use ``log(x, b)``,
    which is essentially short-hand for ``log(x)/log(b)``.

    ``log`` represents the principal branch of the natural
    logarithm. As such it has a branch cut along the negative
    real axis and returns values having a complex argument in
    `(-\\pi, \\pi]`.

    Examples
    ========

    >>> from sympy import log, sqrt, S, I
    >>> log(8, 2)
    3
    >>> log(S(8)/3, 2)
    -log(3)/log(2) + 3
    >>> log(-1 + I*sqrt(3))
    log(2) + 2*I*pi/3

    See Also
    ========

    exp

    """
    args: tTuple[Expr]
    _singularities = (S.Zero, S.ComplexInfinity)

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of the function.
        """
        if argindex == 1:
            return 1 / self.args[0]
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns `e^x`, the inverse function of `\\log(x)`.
        """
        return exp

    @classmethod
    def eval(cls, arg, base=None):
        from sympy.calculus import AccumBounds
        from sympy.sets.setexpr import SetExpr
        arg = sympify(arg)
        if base is not None:
            base = sympify(base)
            if base == 1:
                if arg == 1:
                    return S.NaN
                else:
                    return S.ComplexInfinity
            try:
                n = multiplicity(base, arg)
                if n:
                    return n + log(arg / base ** n) / log(base)
                else:
                    return log(arg) / log(base)
            except ValueError:
                pass
            if base is not S.Exp1:
                return cls(arg) / cls(base)
            else:
                return cls(arg)
        if arg.is_Number:
            if arg.is_zero:
                return S.ComplexInfinity
            elif arg is S.One:
                return S.Zero
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Infinity
            elif arg is S.NaN:
                return S.NaN
            elif arg.is_Rational and arg.p == 1:
                return -cls(arg.q)
        if arg.is_Pow and arg.base is S.Exp1 and arg.exp.is_extended_real:
            return arg.exp
        if isinstance(arg, exp) and arg.exp.is_extended_real:
            return arg.exp
        elif isinstance(arg, exp) and arg.exp.is_number:
            r_, i_ = match_real_imag(arg.exp)
            if i_ and i_.is_comparable:
                i_ %= 2 * pi
                if i_ > pi:
                    i_ -= 2 * pi
                return r_ + expand_mul(i_ * I, deep=False)
        elif isinstance(arg, exp_polar):
            return unpolarify(arg.exp)
        elif isinstance(arg, AccumBounds):
            if arg.min.is_positive:
                return AccumBounds(log(arg.min), log(arg.max))
            elif arg.min.is_zero:
                return AccumBounds(S.NegativeInfinity, log(arg.max))
            else:
                return S.NaN
        elif isinstance(arg, SetExpr):
            return arg._eval_func(cls)
        if arg.is_number:
            if arg.is_negative:
                return pi * I + cls(-arg)
            elif arg is S.ComplexInfinity:
                return S.ComplexInfinity
            elif arg is S.Exp1:
                return S.One
        if arg.is_zero:
            return S.ComplexInfinity
        if not arg.is_Add:
            coeff = arg.as_coefficient(I)
            if coeff is not None:
                if coeff is S.Infinity:
                    return S.Infinity
                elif coeff is S.NegativeInfinity:
                    return S.Infinity
                elif coeff.is_Rational:
                    if coeff.is_nonnegative:
                        return pi * I * S.Half + cls(coeff)
                    else:
                        return -pi * I * S.Half + cls(-coeff)
        if arg.is_number and arg.is_algebraic:
            coeff, arg_ = arg.as_independent(I, as_Add=False)
            if coeff.is_negative:
                coeff *= -1
                arg_ *= -1
            arg_ = expand_mul(arg_, deep=False)
            r_, i_ = arg_.as_independent(I, as_Add=True)
            i_ = i_.as_coefficient(I)
            if coeff.is_real and i_ and i_.is_real and r_.is_real:
                if r_.is_zero:
                    if i_.is_positive:
                        return pi * I * S.Half + cls(coeff * i_)
                    elif i_.is_negative:
                        return -pi * I * S.Half + cls(coeff * -i_)
                else:
                    from sympy.simplify import ratsimp
                    t = (i_ / r_).cancel()
                    t1 = (-t).cancel()
                    atan_table = _log_atan_table()
                    if t in atan_table:
                        modulus = ratsimp(coeff * Abs(arg_))
                        if r_.is_positive:
                            return cls(modulus) + I * atan_table[t]
                        else:
                            return cls(modulus) + I * (atan_table[t] - pi)
                    elif t1 in atan_table:
                        modulus = ratsimp(coeff * Abs(arg_))
                        if r_.is_positive:
                            return cls(modulus) + I * -atan_table[t1]
                        else:
                            return cls(modulus) + I * (pi - atan_table[t1])

    def as_base_exp(self):
        """
        Returns this function in the form (base, exponent).
        """
        return (self, S.One)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Returns the next term in the Taylor series expansion of `\\log(1+x)`.
        """
        from sympy.simplify.powsimp import powsimp
        if n < 0:
            return S.Zero
        x = sympify(x)
        if n == 0:
            return x
        if previous_terms:
            p = previous_terms[-1]
            if p is not None:
                return powsimp(-n * p * x / (n + 1), deep=True, combine='exp')
        return (1 - 2 * (n % 2)) * x ** (n + 1) / (n + 1)

    def _eval_expand_log(self, deep=True, **hints):
        from sympy.concrete import Sum, Product
        force = hints.get('force', False)
        factor = hints.get('factor', False)
        if len(self.args) == 2:
            return expand_log(self.func(*self.args), deep=deep, force=force)
        arg = self.args[0]
        if arg.is_Integer:
            p = perfect_power(arg)
            logarg = None
            coeff = 1
            if p is not False:
                arg, coeff = p
                logarg = self.func(arg)
            if factor:
                p = factorint(arg)
                if arg not in p.keys():
                    logarg = sum((n * log(val) for val, n in p.items()))
            if logarg is not None:
                return coeff * logarg
        elif arg.is_Rational:
            return log(arg.p) - log(arg.q)
        elif arg.is_Mul:
            expr = []
            nonpos = []
            for x in arg.args:
                if force or x.is_positive or x.is_polar:
                    a = self.func(x)
                    if isinstance(a, log):
                        expr.append(self.func(x)._eval_expand_log(**hints))
                    else:
                        expr.append(a)
                elif x.is_negative:
                    a = self.func(-x)
                    expr.append(a)
                    nonpos.append(S.NegativeOne)
                else:
                    nonpos.append(x)
            return Add(*expr) + log(Mul(*nonpos))
        elif arg.is_Pow or isinstance(arg, exp):
            if force or (arg.exp.is_extended_real and (arg.base.is_positive or ((arg.exp + 1).is_positive and (arg.exp - 1).is_nonpositive))) or arg.base.is_polar:
                b = arg.base
                e = arg.exp
                a = self.func(b)
                if isinstance(a, log):
                    return unpolarify(e) * a._eval_expand_log(**hints)
                else:
                    return unpolarify(e) * a
        elif isinstance(arg, Product):
            if force or arg.function.is_positive:
                return Sum(log(arg.function), *arg.limits)
        return self.func(arg)

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.simplify import expand_log, simplify, inversecombine
        if len(self.args) == 2:
            return simplify(self.func(*self.args), **kwargs)
        expr = self.func(simplify(self.args[0], **kwargs))
        if kwargs['inverse']:
            expr = inversecombine(expr)
        expr = expand_log(expr, deep=True)
        return min([expr, self], key=kwargs['measure'])

    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a complex coordinate.

        Examples
        ========

        >>> from sympy import I, log
        >>> from sympy.abc import x
        >>> log(x).as_real_imag()
        (log(Abs(x)), arg(x))
        >>> log(I).as_real_imag()
        (0, pi/2)
        >>> log(1 + I).as_real_imag()
        (log(sqrt(2)), pi/4)
        >>> log(I*x).as_real_imag()
        (log(Abs(x)), arg(I*x))

        """
        sarg = self.args[0]
        if deep:
            sarg = self.args[0].expand(deep, **hints)
        sarg_abs = Abs(sarg)
        if sarg_abs == sarg:
            return (self, S.Zero)
        sarg_arg = arg(sarg)
        if hints.get('log', False):
            hints['complex'] = False
            return (log(sarg_abs).expand(deep, **hints), sarg_arg)
        else:
            return (log(sarg_abs), sarg_arg)

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if (self.args[0] - 1).is_zero:
                return True
            if s.args[0].is_rational and fuzzy_not((self.args[0] - 1).is_zero):
                return False
        else:
            return s.is_rational

    def _eval_is_algebraic(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if (self.args[0] - 1).is_zero:
                return True
            elif fuzzy_not((self.args[0] - 1).is_zero):
                if self.args[0].is_algebraic:
                    return False
        else:
            return s.is_algebraic

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_positive

    def _eval_is_complex(self):
        z = self.args[0]
        return fuzzy_and([z.is_complex, fuzzy_not(z.is_zero)])

    def _eval_is_finite(self):
        arg = self.args[0]
        if arg.is_zero:
            return False
        return arg.is_finite

    def _eval_is_extended_positive(self):
        return (self.args[0] - 1).is_extended_positive

    def _eval_is_zero(self):
        return (self.args[0] - 1).is_zero

    def _eval_is_extended_nonnegative(self):
        return (self.args[0] - 1).is_extended_nonnegative

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        from sympy.simplify.simplify import logcombine
        from sympy.core.symbol import Dummy
        if self.args[0] == x:
            return log(x) if logx is None else logx
        arg = self.args[0]
        t = Dummy('t', positive=True)
        if cdir == 0:
            cdir = 1
        z = arg.subs(x, cdir * t)
        k, l = (Wild('k'), Wild('l'))
        r = z.match(k * t ** l)
        if r is not None:
            k, l = (r[k], r[l])
            if l != 0 and (not l.has(t)) and (not k.has(t)):
                r = l * log(x) if logx is None else l * logx
                r += log(k) - l * log(cdir)
                return r

        def coeff_exp(term, x):
            coeff, exp = (S.One, S.Zero)
            for factor in Mul.make_args(term):
                if factor.has(x):
                    base, exp = factor.as_base_exp()
                    if base != x:
                        try:
                            return term.leadterm(x)
                        except ValueError:
                            return (term, S.Zero)
                else:
                    coeff *= factor
            return (coeff, exp)
        try:
            a, b = z.leadterm(t, logx=logx, cdir=1)
        except (ValueError, NotImplementedError, PoleError):
            s = z._eval_nseries(t, n=n, logx=logx, cdir=1)
            while s.is_Order:
                n += 1
                s = z._eval_nseries(t, n=n, logx=logx, cdir=1)
            try:
                a, b = s.removeO().leadterm(t, cdir=1)
            except ValueError:
                a, b = (s.removeO().as_leading_term(t, cdir=1), S.Zero)
        p = (z / (a * t ** b) - 1)._eval_nseries(t, n=n, logx=logx, cdir=1)
        if p.has(exp):
            p = logcombine(p)
        if isinstance(p, Order):
            n = p.getn()
        _, d = coeff_exp(p, t)
        logx = log(x) if logx is None else logx
        if not d.is_positive:
            res = log(a) - b * log(cdir) + b * logx
            _res = res
            logflags = {'deep': True, 'log': True, 'mul': False, 'power_exp': False, 'power_base': False, 'multinomial': False, 'basic': False, 'force': True, 'factor': False}
            expr = self.expand(**logflags)
            if not a.could_extract_minus_sign() and logx.could_extract_minus_sign():
                _res = _res.subs(-logx, -log(x)).expand(**logflags)
            else:
                _res = _res.subs(logx, log(x)).expand(**logflags)
            if _res == expr:
                return res
            return res + Order(x ** n, x)

        def mul(d1, d2):
            res = {}
            for e1, e2 in product(d1, d2):
                ex = e1 + e2
                if ex < n:
                    res[ex] = res.get(ex, S.Zero) + d1[e1] * d2[e2]
            return res
        pterms = {}
        for term in Add.make_args(p.removeO()):
            co1, e1 = coeff_exp(term, t)
            pterms[e1] = pterms.get(e1, S.Zero) + co1
        k = S.One
        terms = {}
        pk = pterms
        while k * d < n:
            coeff = -S.NegativeOne ** k / k
            for ex in pk:
                _ = terms.get(ex, S.Zero) + coeff * pk[ex]
                terms[ex] = _.nsimplify()
            pk = mul(pk, pterms)
            k += S.One
        res = log(a) - b * log(cdir) + b * logx
        for ex in terms:
            res += terms[ex] * t ** ex
        if a.is_negative and im(z) != 0:
            from sympy.functions.special.delta_functions import Heaviside
            for i, term in enumerate(z.lseries(t)):
                if not term.is_real or i == 5:
                    break
            if i < 5:
                coeff, _ = term.as_coeff_exponent(t)
                res += -2 * I * pi * Heaviside(-im(coeff), 0)
        res = res.subs(t, x / cdir)
        return res + Order(x ** n, x)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg0 = self.args[0].together()
        t = Dummy('t', positive=True)
        if cdir == 0:
            cdir = 1
        z = arg0.subs(x, cdir * t)
        try:
            c, e = z.leadterm(t, logx=logx, cdir=1)
        except ValueError:
            arg = arg0.as_leading_term(x, logx=logx, cdir=cdir)
            return log(arg)
        if c.has(t):
            c = c.subs(t, x / cdir)
            if e != 0:
                raise PoleError('Cannot expand %s around 0' % self)
            return log(c)
        if c == S.One and e == S.Zero:
            return (arg0 - S.One).as_leading_term(x, logx=logx)
        res = log(c) - e * log(cdir)
        logx = log(x) if logx is None else logx
        res += e * logx
        if c.is_negative and im(z) != 0:
            from sympy.functions.special.delta_functions import Heaviside
            for i, term in enumerate(z.lseries(t)):
                if not term.is_real or i == 5:
                    break
            if i < 5:
                coeff, _ = term.as_coeff_exponent(t)
                res += -2 * I * pi * Heaviside(-im(coeff), 0)
        return res