from copy import copy
from functools import reduce
from sympy.polys.agca.ideals import Ideal
from sympy.polys.domains.field import Field
from sympy.polys.orderings import ProductOrder, monomial_key
from sympy.polys.polyerrors import CoercionFailed
from sympy.core.basic import _aresame
from sympy.utilities.iterables import iterable
class SubModulePolyRing(SubModule):
    """
    Submodule of a free module over a generalized polynomial ring.

    Do not instantiate this, use the constructor method of FreeModule instead:

    >>> from sympy.abc import x, y
    >>> from sympy import QQ
    >>> F = QQ.old_poly_ring(x, y).free_module(2)
    >>> F.submodule([x, y], [1, 0])
    <[x, y], [1, 0]>

    Attributes:

    - order - monomial order used
    """

    def __init__(self, gens, container, order='lex', TOP=True):
        SubModule.__init__(self, gens, container)
        if not isinstance(container, FreeModulePolyRing):
            raise NotImplementedError('This implementation is for submodules of ' + 'FreeModulePolyRing, got %s' % container)
        self.order = ModuleOrder(monomial_key(order), self.ring.order, TOP)
        self._gb = None
        self._gbe = None

    def __eq__(self, other):
        if isinstance(other, SubModulePolyRing) and self.order != other.order:
            return False
        return SubModule.__eq__(self, other)

    def _groebner(self, extended=False):
        """Returns a standard basis in sdm form."""
        from sympy.polys.distributedmodules import sdm_groebner, sdm_nf_mora
        if self._gbe is None and extended:
            gb, gbe = sdm_groebner([self.ring._vector_to_sdm(x, self.order) for x in self.gens], sdm_nf_mora, self.order, self.ring.dom, extended=True)
            self._gb, self._gbe = (tuple(gb), tuple(gbe))
        if self._gb is None:
            self._gb = tuple(sdm_groebner([self.ring._vector_to_sdm(x, self.order) for x in self.gens], sdm_nf_mora, self.order, self.ring.dom))
        if extended:
            return (self._gb, self._gbe)
        else:
            return self._gb

    def _groebner_vec(self, extended=False):
        """Returns a standard basis in element form."""
        if not extended:
            return [FreeModuleElement(self, tuple(self.ring._sdm_to_vector(x, self.rank))) for x in self._groebner()]
        gb, gbe = self._groebner(extended=True)
        return ([self.convert(self.ring._sdm_to_vector(x, self.rank)) for x in gb], [self.ring._sdm_to_vector(x, len(self.gens)) for x in gbe])

    def _contains(self, x):
        from sympy.polys.distributedmodules import sdm_zero, sdm_nf_mora
        return sdm_nf_mora(self.ring._vector_to_sdm(x, self.order), self._groebner(), self.order, self.ring.dom) == sdm_zero()

    def _syzygies(self):
        """Compute syzygies. See [SCA, algorithm 2.5.4]."""
        k = len(self.gens)
        r = self.rank
        zero = self.ring.convert(0)
        one = self.ring.convert(1)
        Rkr = self.ring.free_module(r + k)
        newgens = []
        for j, f in enumerate(self.gens):
            m = [0] * (r + k)
            for i, v in enumerate(f):
                m[i] = f[i]
            for i in range(k):
                m[r + i] = one if j == i else zero
            m = FreeModuleElement(Rkr, tuple(m))
            newgens.append(m)
        F = Rkr.submodule(*newgens, order='ilex', TOP=False)
        G = F._groebner_vec()
        G0 = [x[r:] for x in G if all((y == zero for y in x[:r]))]
        return G0

    def _in_terms_of_generators(self, e):
        """Expression in terms of generators. See [SCA, 2.8.1]."""
        M = self.ring.free_module(self.rank).submodule(*(e,) + self.gens)
        S = M.syzygy_module(order='ilex', TOP=False)
        G = S._groebner_vec()
        e = [x for x in G if self.ring.is_unit(x[0])][0]
        return [-x / e[0] for x in e[1:]]

    def reduce_element(self, x, NF=None):
        """
        Reduce the element ``x`` of our container modulo ``self``.

        This applies the normal form ``NF`` to ``x``. If ``NF`` is passed
        as none, the default Mora normal form is used (which is not unique!).
        """
        from sympy.polys.distributedmodules import sdm_nf_mora
        if NF is None:
            NF = sdm_nf_mora
        return self.container.convert(self.ring._sdm_to_vector(NF(self.ring._vector_to_sdm(x, self.order), self._groebner(), self.order, self.ring.dom), self.rank))

    def _intersect(self, other, relations=False):
        fi = self.gens
        hi = other.gens
        r = self.rank
        ci = [[0] * (2 * r) for _ in range(r)]
        for k in range(r):
            ci[k][k] = 1
            ci[k][r + k] = 1
        di = [list(f) + [0] * r for f in fi]
        ei = [[0] * r + list(h) for h in hi]
        syz = self.ring.free_module(2 * r).submodule(*ci + di + ei)._syzygies()
        nonzero = [x for x in syz if any((y != self.ring.zero for y in x[:r]))]
        res = self.container.submodule(*([-y for y in x[:r]] for x in nonzero))
        reln1 = [x[r:r + len(fi)] for x in nonzero]
        reln2 = [x[r + len(fi):] for x in nonzero]
        if relations:
            return (res, reln1, reln2)
        return res

    def _module_quotient(self, other, relations=False):
        if relations and len(other.gens) != 1:
            raise NotImplementedError
        if len(other.gens) == 0:
            return self.ring.ideal(1)
        elif len(other.gens) == 1:
            g1 = list(other.gens[0]) + [1]
            gi = [list(x) + [0] for x in self.gens]
            M = self.ring.free_module(self.rank + 1).submodule(*[g1] + gi, order='ilex', TOP=False)
            if not relations:
                return self.ring.ideal(*[x[-1] for x in M._groebner_vec() if all((y == self.ring.zero for y in x[:-1]))])
            else:
                G, R = M._groebner_vec(extended=True)
                indices = [i for i, x in enumerate(G) if all((y == self.ring.zero for y in x[:-1]))]
                return (self.ring.ideal(*[G[i][-1] for i in indices]), [[-x for x in R[i][1:]] for i in indices])
        return reduce(lambda x, y: x.intersect(y), (self._module_quotient(self.container.submodule(x)) for x in other.gens))