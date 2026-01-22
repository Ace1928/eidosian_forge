from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
class ModuleImplementedIdeal(Ideal):
    """
    Ideal implementation relying on the modules code.

    Attributes:

    - _module - the underlying module
    """

    def __init__(self, ring, module):
        Ideal.__init__(self, ring)
        self._module = module

    def _contains_elem(self, x):
        return self._module.contains([x])

    def _contains_ideal(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self._module.is_submodule(J._module)

    def _intersect(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self.__class__(self.ring, self._module.intersect(J._module))

    def _quotient(self, J, **opts):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self._module.module_quotient(J._module, **opts)

    def _union(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self.__class__(self.ring, self._module.union(J._module))

    @property
    def gens(self):
        """
        Return generators for ``self``.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x, y
        >>> list(QQ.old_poly_ring(x, y).ideal(x, y, x**2 + y).gens)
        [x, y, x**2 + y]
        """
        return (x[0] for x in self._module.gens)

    def is_zero(self):
        """
        Return True if ``self`` is the zero ideal.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).ideal(x).is_zero()
        False
        >>> QQ.old_poly_ring(x).ideal().is_zero()
        True
        """
        return self._module.is_zero()

    def is_whole_ring(self):
        """
        Return True if ``self`` is the whole ring, i.e. one generator is a unit.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ, ilex
        >>> QQ.old_poly_ring(x).ideal(x).is_whole_ring()
        False
        >>> QQ.old_poly_ring(x).ideal(3).is_whole_ring()
        True
        >>> QQ.old_poly_ring(x, order=ilex).ideal(2 + x).is_whole_ring()
        True
        """
        return self._module.is_full_module()

    def __repr__(self):
        from sympy.printing.str import sstr
        return '<' + ','.join((sstr(x) for [x] in self._module.gens)) + '>'

    def _product(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self.__class__(self.ring, self._module.submodule(*[[x * y] for [x] in self._module.gens for [y] in J._module.gens]))

    def in_terms_of_generators(self, e):
        """
        Express ``e`` in terms of the generators of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x**2 + 1, x)
        >>> I.in_terms_of_generators(1)
        [1, -x]
        """
        return self._module.in_terms_of_generators([e])

    def reduce_element(self, x, **options):
        return self._module.reduce_element([x], **options)[0]