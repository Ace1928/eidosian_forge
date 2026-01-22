from itertools import combinations_with_replacement, product
from textwrap import dedent
from sympy.core import Mul, S, Tuple, sympify
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import PicklableWithSlots, dict_from_expr
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence, iterable
class MonomialOps:
    """Code generator of fast monomial arithmetic functions. """

    def __init__(self, ngens):
        self.ngens = ngens

    def _build(self, code, name):
        ns = {}
        exec(code, ns)
        return ns[name]

    def _vars(self, name):
        return ['%s%s' % (name, i) for i in range(self.ngens)]

    def mul(self):
        name = 'monomial_mul'
        template = dedent('        def %(name)s(A, B):\n            (%(A)s,) = A\n            (%(B)s,) = B\n            return (%(AB)s,)\n        ')
        A = self._vars('a')
        B = self._vars('b')
        AB = ['%s + %s' % (a, b) for a, b in zip(A, B)]
        code = template % {'name': name, 'A': ', '.join(A), 'B': ', '.join(B), 'AB': ', '.join(AB)}
        return self._build(code, name)

    def pow(self):
        name = 'monomial_pow'
        template = dedent('        def %(name)s(A, k):\n            (%(A)s,) = A\n            return (%(Ak)s,)\n        ')
        A = self._vars('a')
        Ak = ['%s*k' % a for a in A]
        code = template % {'name': name, 'A': ', '.join(A), 'Ak': ', '.join(Ak)}
        return self._build(code, name)

    def mulpow(self):
        name = 'monomial_mulpow'
        template = dedent('        def %(name)s(A, B, k):\n            (%(A)s,) = A\n            (%(B)s,) = B\n            return (%(ABk)s,)\n        ')
        A = self._vars('a')
        B = self._vars('b')
        ABk = ['%s + %s*k' % (a, b) for a, b in zip(A, B)]
        code = template % {'name': name, 'A': ', '.join(A), 'B': ', '.join(B), 'ABk': ', '.join(ABk)}
        return self._build(code, name)

    def ldiv(self):
        name = 'monomial_ldiv'
        template = dedent('        def %(name)s(A, B):\n            (%(A)s,) = A\n            (%(B)s,) = B\n            return (%(AB)s,)\n        ')
        A = self._vars('a')
        B = self._vars('b')
        AB = ['%s - %s' % (a, b) for a, b in zip(A, B)]
        code = template % {'name': name, 'A': ', '.join(A), 'B': ', '.join(B), 'AB': ', '.join(AB)}
        return self._build(code, name)

    def div(self):
        name = 'monomial_div'
        template = dedent('        def %(name)s(A, B):\n            (%(A)s,) = A\n            (%(B)s,) = B\n            %(RAB)s\n            return (%(R)s,)\n        ')
        A = self._vars('a')
        B = self._vars('b')
        RAB = ['r%(i)s = a%(i)s - b%(i)s\n    if r%(i)s < 0: return None' % {'i': i} for i in range(self.ngens)]
        R = self._vars('r')
        code = template % {'name': name, 'A': ', '.join(A), 'B': ', '.join(B), 'RAB': '\n    '.join(RAB), 'R': ', '.join(R)}
        return self._build(code, name)

    def lcm(self):
        name = 'monomial_lcm'
        template = dedent('        def %(name)s(A, B):\n            (%(A)s,) = A\n            (%(B)s,) = B\n            return (%(AB)s,)\n        ')
        A = self._vars('a')
        B = self._vars('b')
        AB = ['%s if %s >= %s else %s' % (a, a, b, b) for a, b in zip(A, B)]
        code = template % {'name': name, 'A': ', '.join(A), 'B': ', '.join(B), 'AB': ', '.join(AB)}
        return self._build(code, name)

    def gcd(self):
        name = 'monomial_gcd'
        template = dedent('        def %(name)s(A, B):\n            (%(A)s,) = A\n            (%(B)s,) = B\n            return (%(AB)s,)\n        ')
        A = self._vars('a')
        B = self._vars('b')
        AB = ['%s if %s <= %s else %s' % (a, a, b, b) for a, b in zip(A, B)]
        code = template % {'name': name, 'A': ', '.join(A), 'B': ', '.join(B), 'AB': ', '.join(AB)}
        return self._build(code, name)