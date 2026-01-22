from collections import OrderedDict, defaultdict
from functools import reduce
from itertools import chain, product
from operator import mul, add
import copy
import math
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util._expr import Expr
from .util.periodic import mass_from_composition
from .util.parsing import (
from .units import default_units, is_quantity, unit_of, to_unitless
from ._util import intdiv
from .util.pyutil import deprecated, DeferredImport, ChemPyDeprecationWarning
class Equilibrium(Reaction):
    """Represents an equilibrium reaction

    See :class:`Reaction` for parameters

    """
    _str_arrow = '='
    param_char = 'K'

    def check_consistent_units(self, throw=False):
        if is_quantity(self.param):
            exponent = sum(self.prod.values()) - sum(self.reac.values())
            unit_param = unit_of(self.param, simplified=True)
            unit_expected = unit_of(default_units.molar ** exponent, simplified=True)
            if unit_param == unit_expected:
                return True
            elif throw:
                raise ValueError('Inconsistent units in equilibrium: %s vs %s' % (unit_param, unit_expected))
            else:
                return False
        else:
            return True

    def as_reactions(self, kf=None, kb=None, units=None, variables=None, backend=math, new_name=None, **kwargs):
        """Creates a forward and backward :class:`Reaction` pair

        Parameters
        ----------
        kf : float or RateExpr
        kb : float or RateExpr
        units : module
        variables : dict, optional
        backend : module

        """
        nb = sum(self.prod.values())
        nf = sum(self.reac.values())
        if units is None:
            if hasattr(kf, 'units') or hasattr(kb, 'units'):
                raise ValueError('units missing')
            c0 = 1
        else:
            c0 = 1 * units.molar
        if kf is None:
            fw_name = self.name
            bw_name = new_name
            if kb is None:
                try:
                    kf, kb = self.param
                except TypeError:
                    raise ValueError('Exactly one rate needs to be provided')
            else:
                kf = kb * self.param * c0 ** (nb - nf)
        elif kb is None:
            kb = kf / (self.param * c0 ** (nb - nf))
            fw_name = new_name
            bw_name = self.name
        else:
            raise ValueError('Exactly one rate needs to be provided')
        return (Reaction(self.reac, self.prod, kf, self.inact_reac, self.inact_prod, ref=self.ref, name=fw_name, **kwargs), Reaction(self.prod, self.reac, kb, self.inact_prod, self.inact_reac, ref=self.ref, name=bw_name, **kwargs))

    def equilibrium_expr(self):
        """Turns self.param into a :class:`EqExpr` instance (if not already)

        Examples
        --------
        >>> r = Equilibrium.from_string('2 A + B = 3 C; 7')
        >>> eqex = r.equilibrium_expr()
        >>> eqex.args[0] == 7
        True

        """
        from .util._expr import Expr
        from .thermodynamics import MassActionEq
        if isinstance(self.param, Expr):
            return self.param
        else:
            try:
                convertible = self.param.as_EqExpr
            except AttributeError:
                return MassActionEq([self.param])
            else:
                return convertible()

    def equilibrium_constant(self, variables=None, backend=math):
        """Return equilibrium constant

        Parameters
        ----------
        variables : dict, optional
        backend : module, optional

        """
        return self.equilibrium_expr().eq_const(variables, backend=backend)

    def equilibrium_equation(self, variables, backend=None, **kwargs):
        return self.equilibrium_expr().equilibrium_equation(variables, equilibrium=self, backend=backend, **kwargs)

    @deprecated(use_instead=equilibrium_constant)
    def K(self, *args, **kwargs):
        return self.equilibrium_constant(*args, **kwargs)

    def Q(self, substances, concs):
        """Calculates the equilibrium qoutient"""
        stoich = self.non_precipitate_stoich(substances)
        return equilibrium_quotient(concs, stoich)

    def precipitate_factor(self, substances, sc_concs):
        factor = 1
        for r, n in self.reac.items():
            if r.precipitate:
                factor *= sc_concs[substances.index(r)] ** (-n)
        for p, n in self.prod.items():
            if p.precipitate:
                factor *= sc_concs[substances.index(p)] ** n
        return factor

    def dimensionality(self, substances):
        result = 0
        for r, n in self.reac.items():
            if getattr(substances[r], 'phase_idx', 0) > 0:
                continue
            result -= n
        for p, n in self.prod.items():
            if getattr(substances[p], 'phase_idx', 0) > 0:
                continue
            result += n
        return result

    def __rmul__(self, other):
        try:
            other_is_int = other.is_integer
        except AttributeError:
            other_is_int = isinstance(other, int)
        if not other_is_int or not isinstance(self, Equilibrium):
            return NotImplemented
        param = None if self.param is None else self.param ** other
        if other < 0:
            other *= -1
            flip = True
        else:
            flip = False
        other = int(other)
        reac = dict(other * ArithmeticDict(int, self.reac))
        prod = dict(other * ArithmeticDict(int, self.prod))
        inact_reac = dict(other * ArithmeticDict(int, self.inact_reac))
        inact_prod = dict(other * ArithmeticDict(int, self.inact_prod))
        if flip:
            reac, prod = (prod, reac)
            inact_reac, inact_prod = (inact_prod, inact_reac)
        return Equilibrium(reac, prod, param, inact_reac=inact_reac, inact_prod=inact_prod)

    def __neg__(self):
        return -1 * self

    def __mul__(self, other):
        return other * self

    def __add__(self, other):
        keys = set()
        for key in chain(self.reac.keys(), self.prod.keys(), other.reac.keys(), other.prod.keys()):
            keys.add(key)
        reac, prod = ({}, {})
        for key in keys:
            n = self.prod.get(key, 0) - self.reac.get(key, 0) + other.prod.get(key, 0) - other.reac.get(key, 0)
            if n < 0:
                reac[key] = -n
            elif n > 0:
                prod[key] = n
            else:
                pass
        if (self.param, other.param) == (None, None):
            param = None
        else:
            param = self.param * other.param
        return Equilibrium(reac, prod, param)

    def __sub__(self, other):
        return self + -1 * other

    @staticmethod
    def eliminate(rxns, wrt):
        """Linear combination coefficients for elimination of a substance

        Parameters
        ----------
        rxns : iterable of Equilibrium instances
        wrt : str (substance key)

        Examples
        --------
        >>> e1 = Equilibrium({'Cd+2': 4, 'H2O': 4}, {'Cd4(OH)4+4': 1, 'H+': 4}, 10**-32.5)
        >>> e2 = Equilibrium({'Cd(OH)2(s)': 1}, {'Cd+2': 1, 'OH-': 2}, 10**-14.4)
        >>> Equilibrium.eliminate([e1, e2], 'Cd+2')
        [1, 4]
        >>> print(1*e1 + 4*e2)
        4 Cd(OH)2(s) + 4 H2O = Cd4(OH)4+4 + 4 H+ + 8 OH-; 7.94e-91

        """
        import sympy
        viol = [r.net_stoich([wrt])[0] for r in rxns]
        factors = defaultdict(int)
        for v in viol:
            for f in sympy.primefactors(v):
                factors[f] = max(factors[f], sympy.Abs(v // f))
        rcd = reduce(mul, (k ** v for k, v in factors.items()))
        viol[0] *= -1
        return [rcd // v for v in viol]

    def cancel(self, rxn):
        """Multiplier of how many times rxn can be added/subtracted.

        Parameters
        ----------
        rxn : Equilibrium

        Examples
        --------
        >>> e1 = Equilibrium({'Cd(OH)2(s)': 4, 'H2O': 4},
        ...                  {'Cd4(OH)4+4': 1, 'H+': 4, 'OH-': 8}, 7.94e-91)
        >>> e2 = Equilibrium({'H2O': 1}, {'H+': 1, 'OH-': 1}, 10**-14)
        >>> e1.cancel(e2)
        -4
        >>> print(e1 - 4*e2)
        4 Cd(OH)2(s) = Cd4(OH)4+4 + 4 OH-; 7.94e-35

        """
        keys = rxn.keys()
        s1 = self.net_stoich(keys)
        s2 = rxn.net_stoich(keys)
        candidate = float('inf')
        for v1, v2 in zip(s1, s2):
            r = intdiv(-v1, v2)
            candidate = min(candidate, r, key=abs)
        return candidate