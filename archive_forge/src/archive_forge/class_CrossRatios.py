from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
class CrossRatios(dict):
    """
    Represents assigned shape parameters/cross ratios as
    dictionary. The cross ratios are according to SnapPy convention, so we
    have::

        z = 1 - 1/zp, zp = 1 - 1/zpp, zpp = 1 - 1/z

    where::

        z   is at the edge 01 and equal to s0 * s1 * (c_1010 * c_0101) / (c_1001 * c_0110)
        zp  is at the edge 02 and equal to s0 * s2 * (c_1001 * c_0110) / (c_1100 * c_0011)
        zpp is at the edge 03 and equal to s0 * s3 * (c_1100 * c_0011) / (c_0101 * c_1010).

    Note that this is different from the convention used in
    Garoufalidis, Goerner, Zickert:
    Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
    http://arxiv.org/abs/1207.6711
    """

    def __init__(self, d, is_numerical=True, manifold_thunk=None):
        super(CrossRatios, self).__init__(d)
        self._is_numerical = is_numerical
        self._manifold_thunk = manifold_thunk
        self._edge_cache = {}
        self._matrix_cache = []
        self._inverse_matrix_cache = []
        self.dimension = 0

    @staticmethod
    def from_snappy_manifold(M, dec_prec=None, bits_prec=None, intervals=False):
        """
        Constructs an assignment of shape parameters/cross ratios using
        the tetrahehdra_shapes method of a given SnapPy manifold. The optional
        parameters are the same as that of tetrahedra_shapes.
        """
        shapes = M.tetrahedra_shapes('rect', dec_prec=dec_prec, bits_prec=bits_prec, intervals=intervals)
        d = {}
        for i, shape in enumerate(shapes):
            d['z_0000_%d' % i] = shape
            d['zp_0000_%d' % i] = 1 / (1 - shape)
            d['zpp_0000_%d' % i] = 1 - 1 / shape
        return CrossRatios(d, is_numerical=True, manifold_thunk=lambda M=M: M)

    def __repr__(self):
        dict_repr = dict.__repr__(self)
        return 'CrossRatios(%s, is_numerical = %r, ...)' % (dict_repr, self._is_numerical)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text('CrossRatios(...)')
        else:
            with p.group(4, 'CrossRatios(', ')'):
                p.breakable()
                p.pretty(dict(self))
                p.text(',')
                p.breakable()
                p.text('is_numerical = %r, ...' % self._is_numerical)

    def get_manifold(self):
        """
        Get the manifold for which this structure represents a solution
        to the gluing equations.
        """
        return self._manifold_thunk()

    def num_tetrahedra(self):
        """
        The number of tetrahedra for which we have cross ratios.
        """
        return _num_tetrahedra(self)

    def N(self):
        """
        Get the N such that these cross ratios are for
        SL/PSL(N,C)-representations.
        """
        return _N_for_shapes(self)

    def numerical(self):
        """
        Turn exact solutions into numerical solutions using pari. Similar to
        numerical() of PtolemyCoordinates. See help(ptolemy.PtolemyCoordinates)
        for example.
        """
        if self._is_numerical:
            return self
        return ZeroDimensionalComponent([CrossRatios(d, is_numerical=True, manifold_thunk=self._manifold_thunk) for d in _to_numerical(self)])

    def to_PUR(self):
        """
        If any Ptolemy coordinates are given as Rational Univariate
        Representation, convert them to Polynomial Univariate Representation and
        return the result.

        See to_PUR of RUR.

        This conversion might lead to very large coefficients.
        """
        return CrossRatios(_apply_to_RURs(self, RUR.to_PUR), is_numerical=self._is_numerical, manifold_thunk=self._manifold_thunk)

    def multiply_terms_in_RUR(self):
        """
        If a cross ratio is given as Rational Univariate Representation
        with numerator and denominator being a product, multiply the terms and
        return the result.

        See multiply_terms of RUR.

        This loses information about how the numerator and denominator are
        factorised.
        """
        return CrossRatios(_apply_to_RURs(self, RUR.multiply_terms), is_numerical=self._is_numerical, manifold_thunk=self._manifold_thunk)

    def multiply_and_simplify_terms_in_RUR(self):
        """
        If a cross ratio is given as Rational Univariate Representation
        with numerator and denominator being a product, multiply the terms,
        reduce the fraction and return the result.

        See multiply_and_simplify_terms of RUR.

        This loses information about how the numerator and denominator are
        factorised.

        """
        return CrossRatios(_apply_to_RURs(self, RUR.multiply_and_simplify_terms), is_numerical=self._is_numerical, manifold_thunk=self._manifold_thunk)

    def volume_numerical(self, drop_negative_vols=False):
        """
        Turn into (Galois conjugate) numerical solutions and compute volumes.
        If already numerical, only compute the one volume.
        See numerical().

        If drop_negative_vols = True is given as optional argument,
        only return non-negative volumes.
        """
        if self._is_numerical:
            return sum([_volume(z) for key, z in list(self.items()) if 'z_' in key])
        else:
            vols = ZeroDimensionalComponent([num.volume_numerical() for num in self.numerical()])
            if drop_negative_vols:
                return [vol for vol in vols if vol > -1e-12]
            return vols

    @staticmethod
    def _cyclic_three_perm_sign(v0, v1, v2):
        """
        Returns +1 or -1. It is +1 if and only if (v0, v1, v2) is in the
        orbit of (0, 1, 2) under the A4-action.
        """
        for t in [(v0, v1, v2), (v1, v2, v0), (v2, v0, v1)]:
            if t in [(0, 1, 2), (1, 3, 2), (2, 3, 0), (3, 1, 0)]:
                return +1
        return -1

    def _shape_at_tet_point_and_edge(self, tet, pt, edge):
        """
        Given the index of a tetrahedron and two quadruples (any iterabel) of
        integers, give the cross ratio at that integral point and edge of that
        tetrahedron.
        This method translates the SnapPy conventions of labeling simplices
        and the conventions in Definition 4.2 of

        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711
        """
        postfix = '_%d%d%d%d' % tuple(pt) + '_%d' % tet
        if tuple(edge) in [(1, 1, 0, 0), (0, 0, 1, 1)]:
            return self['z' + postfix]
        if tuple(edge) in [(1, 0, 1, 0), (0, 1, 0, 1)]:
            return self['zp' + postfix]
        if tuple(edge) in [(1, 0, 0, 1), (0, 1, 1, 0)]:
            return self['zpp' + postfix]
        raise Exception('Invalid edge ' + str(edge))

    def x_coordinate(self, tet, pt):
        """
        Returns the X-coordinate for the tetrahedron with index tet
        at the point pt (quadruple of integers adding up to N).

        See Definition 10.9:
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711
        """
        result = 1
        for v0 in range(4):
            for v1 in range(v0 + 1, 4):
                e = [_kronecker_delta(v0, i) + _kronecker_delta(v1, i) for i in range(4)]
                p = [x1 - x2 for x1, x2 in zip(pt, e)]
                if all((x >= 0 for x in p)):
                    result *= self._shape_at_tet_point_and_edge(tet, p, e)
        return -result

    def _get_identity_matrix(self):
        N = self.N()
        return [[_kronecker_delta(i, j) for i in range(N)] for j in range(N)]

    def long_edge(self, tet, v0, v1, v2):
        """
        The matrix that labels a long edge starting at vertex (v0, v1, v2)
        of a doubly truncated simplex corresponding to the ideal tetrahedron
        with index tet.

        This matrix was labeled alpha^{v0v1v2} in Figure 18 of
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711

        It is computed using equation 10.22.

        The resulting matrix is given as a python list of lists.
        """
        key = 'long_edge'
        if key not in self._edge_cache:
            N = self.N()
            m = [[_kronecker_delta(i + j, N - 1) for i in range(N)] for j in range(N)]
            self._edge_cache[key] = m
        return self._edge_cache[key]

    def middle_edge(self, tet, v0, v1, v2):
        """
        The matrix that labels a middle edge starting at vertex (v0, v1, v2)
        of a doubly truncated simplex corresponding to the ideal tetrahedron
        with index tet.

        This matrix was labeled beta^{v0v1v2} in Figure 18 of
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711

        It is computed using equation 10.22.

        The resulting matrix is given as a python list of lists.
        """
        key = 'middle_%d_%d%d%d' % (tet, v0, v1, v2)
        if key not in self._edge_cache:
            N = self.N()
            sgn = CrossRatios._cyclic_three_perm_sign(v0, v1, v2)
            m = self._get_identity_matrix()
            for k in range(1, N):
                prod1 = self._get_identity_matrix()
                for i in range(1, N - k + 1):
                    prod1 = matrix.matrix_mult(prod1, _X(N, i, 1))
                prod2 = self._get_identity_matrix()
                for i in range(1, N - k):
                    pt = [k * _kronecker_delta(v2, j) + i * _kronecker_delta(v0, j) + (N - k - i) * _kronecker_delta(v1, j) for j in range(4)]
                    prod2 = matrix.matrix_mult(prod2, _H(N, i, self.x_coordinate(tet, pt) ** (-sgn)))
                m = matrix.matrix_mult(m, matrix.matrix_mult(prod1, prod2))
            dpm = [[-(-1) ** (N - i) * _kronecker_delta(i, j) for i in range(N)] for j in range(N)]
            m = matrix.matrix_mult(m, dpm)
            self._edge_cache[key] = m
        return self._edge_cache[key]

    def short_edge(self, tet, v0, v1, v2):
        """
        The matrix that labels a long edge starting at vertex (v0, v1, v2)
        of a doubly truncated simplex corresponding to the ideal tetrahedron
        with index tet.

        This matrix was labeled gamma^{v0v1v2} in Figure 18 of
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711

        It is computed using equation 10.22.

        The resulting matrix is given as a python list of lists.
        """
        key = 'short_%d_%d%d%d' % (tet, v0, v1, v2)
        if key not in self._edge_cache:
            edge = [_kronecker_delta(v0, i) + _kronecker_delta(v1, i) for i in range(4)]
            sgn = CrossRatios._cyclic_three_perm_sign(v0, v1, v2)
            N = self.N()
            m = self._get_identity_matrix()
            for a0 in range(N - 1):
                a1 = N - 2 - a0
                pt = [a0 * _kronecker_delta(v0, i) + a1 * _kronecker_delta(v1, i) for i in range(4)]
                cross_ratio = self._shape_at_tet_point_and_edge(tet, pt, edge)
                m = matrix.matrix_mult(m, _H(N, a0 + 1, cross_ratio ** sgn))
            self._edge_cache[key] = m
        return self._edge_cache[key]

    def _init_matrix_and_inverse_cache(self):
        if self._matrix_cache and self._inverse_matrix_cache:
            return
        self._matrix_cache, self._inverse_matrix_cache = findLoops.images_of_original_generators(self, penalties=(0, 1, 1))

    def evaluate_word(self, word, G=None):
        """
        Given a word in the generators of the fundamental group,
        compute the corresponding matrix. By default, these are the
        generators of the unsimplified presentation of the fundamental
        group. An optional SnapPy fundamental group can be given if the
        words are in generators of a different presentation, e.g.,
        c.evaluate_word(word, M.fundamental_group(True)) to
        evaluate a word in the simplified presentation returned by
        M.fundamental_group(True).

        For now, the matrix is returned as list of lists.
        """
        self._init_matrix_and_inverse_cache()
        return findLoops.evaluate_word(self._get_identity_matrix(), self._matrix_cache, self._inverse_matrix_cache, word, G)

    def check_against_manifold(self, M=None, epsilon=None):
        """
        Checks that the given solution really is a solution to the PGL(N,C) gluing
        equations of a manifold. Usage similar to check_against_manifold of
        PtolemyCoordinates. See help(ptolemy.PtolemtyCoordinates) for example.

        === Arguments ===

        M --- manifold to check this for
        epsilon --- maximal allowed error when checking the relations, use
        None for exact comparison.
        """
        if M is None:
            M = self.get_manifold()
        if M is None:
            raise Exception('Need to give manifold')
        some_z = list(self.keys())[0]
        variable_name, index, tet_index = some_z.split('_')
        if variable_name not in ['z', 'zp', 'zpp']:
            raise Exception('Variable not z, zp, or, zpp')
        if len(index) != 4:
            raise Exception('Not 4 indices')
        N = sum([int(x) for x in index]) + 2
        matrix_with_explanations = M.gluing_equations_pgl(N, equation_type='all')
        matrix = matrix_with_explanations.matrix
        rows = matrix_with_explanations.explain_rows
        cols = matrix_with_explanations.explain_columns
        for row in range(len(rows)):
            product = 1
            for col in range(len(cols)):
                cross_ratio_variable = cols[col]
                cross_ratio_value = self[cross_ratio_variable]
                product = product * cross_ratio_value ** matrix[row, col]
            _check_relation(product - 1, epsilon, 'Gluing equation %s' % rows[row])

    def induced_representation(self, N):
        """
        Given a PSL(2,C) representation constructs the induced representation
        for the given N.
        The induced representation is in SL(N,C) if N is odd and
        SL(N,C) / {+1,-1} if N is even and is described in the Introduction of
        Garoufalidis, Thurston, Zickert
        The Complex Volume of SL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1111.2828

        There is a canonical group homomorphism SL(2,C)->SL(N,C) coming from
        the the natural SL(2,C)-action on the vector space Sym^{N-1}(C^2).
        This homomorphisms decends to a homomorphism from PSL(2,C) if one
        divides the right side by {+1,-1} when N is even.
        Composing a representation with this homomorphism gives the induced
        representation.
        """
        num_tetrahedra = self.num_tetrahedra()
        if self.N() != 2:
            raise Exception('Cross ratios need to come from a PSL(2,C) representation')

        def key_value_pair(v, t, index):
            new_key = v + '_%d%d%d%d' % tuple(index) + '_%d' % t
            old_key = v + '_0000' + '_%d' % t
            return (new_key, self[old_key])
        d = dict([key_value_pair(v, t, index) for v in ['z', 'zp', 'zpp'] for t in range(num_tetrahedra) for index in utilities.quadruples_with_fixed_sum_iterator(N - 2)])
        return CrossRatios(d, is_numerical=self._is_numerical, manifold_thunk=self._manifold_thunk)

    def is_real(self, epsilon):
        """
        Returns True if all cross ratios are real (have absolute imaginary
        part < epsilon where epsilon is given as argument).
        This means that the corresponding representation is in PSL(N,R).
        """
        if not self._is_numerical:
            raise NumericalMethodError('is_real')
        for v in self.values():
            if v.imag().abs() > epsilon:
                return False
        return True

    def is_induced_from_psl2(self, epsilon=None):
        """
        For each simplex and each edges, checks that all cross ratios of that
        simplex that are parallel to that each are the same (maximal absolute
        difference is the epsilon given as argument).
        This means that the corresponding representation is induced by a
        PSL(2,C) representation.
        """
        d = {}
        for key, value in self.items():
            variable_name, index, tet_index = key.split('_')
            if variable_name not in ['z', 'zp', 'zpp']:
                raise Exception('Variable not z, zp, or, zpp')
            if len(index) != 4:
                raise Exception('Not 4 indices')
            short_key = variable_name + '_' + tet_index
            old_value = d.setdefault(short_key, value)
            if epsilon is None:
                if value != old_value:
                    return False
            elif (value - old_value).abs() > epsilon:
                return False
        return True

    def is_pu_2_1_representation(self, epsilon, epsilon2=None):
        """
        Returns True if the representation is also a
        PU(2,1)-representation. This uses Proposition 3.5 and the
        remark following that proposition in [FKR2013]_.

        If a condition given in that Proposition is violated, the method returns
        an object whose Boolean value is still False and that indicates which condition
        was violated. Thus, this method can still be used in ``if`` statements.

        The method tests the following complex equalities and inequalities:

        * the three complex equations given in (3.5.1) of [FKR2013]_.
        * the inequality z\\ :sub:`ijl` :math:`\\\\not=` -1.

        **Remark:** It does not check whether all z\\ :sub:`ij` * z\\ :sub:`ji` are real or
        not as these are still valid CR configurations, see the remark following
        Proposition 3.5.

        The user has to supply an epsilon: an equality/inequality is considered
        to be true if and only if the absolute value | LHS - RHS | of difference between the
        left and right hand side is less/greater than epsilon.

        The user can supply another parameter, epsilon2. If any | LHS - RHS | is in
        the interval [epsilon, epsilon2], this method fails with an exception
        as the value of | LHS - RHS | is an ambiguous interval where
        it is unclear whether inequality fails to hold because it truly does
        hold or just because of numerical noise.
        """

        def is_zero(val):
            if val.abs() < epsilon:
                return True
            if epsilon2:
                if not epsilon2 < val.abs():
                    raise Exception('Ambiguous error when determining whether a condition was fulfilled or nor: %s' % val)
            return False

        def mainCondition(key_zij, key_zji, key_zkl, key_zlk):
            lhs = self[key_zij] * self[key_zji]
            rhs = (self[key_zkl] * self[key_zlk]).conj()
            if not is_zero(lhs - rhs):
                reason = '%s * %s = conjugate(%s * %s) not fulfilled' % (key_zij, key_zji, key_zkl, key_zlk)
                return NotPU21Representation(reason)
            return True

        def tripleRatioCondition(key_zji, key_zki, key_zli):
            tripleRatio = self[key_zji] * self[key_zki] * self[key_zli]
            if is_zero(tripleRatio - 1):
                reason = 'Triple ratio %s * %s * %s = 1' % (key_zji, key_zki, key_zli)
                return NotPU21Representation(reason)
            return True
        if self.N() != 3:
            raise Exception('PU(2,1)-representations only allowed for N = 3')
        if not self._is_numerical:
            raise NumericalMethodError('is_pu_2_1_representation')
        for t in range(self.num_tetrahedra()):
            m0 = mainCondition('z_1000_%d' % t, 'z_0100_%d' % t, 'z_0010_%d' % t, 'z_0001_%d' % t)
            if not m0:
                return m0
            m1 = mainCondition('zp_1000_%d' % t, 'zp_0010_%d' % t, 'zp_0100_%d' % t, 'zp_0001_%d' % t)
            if not m1:
                return m1
            m2 = mainCondition('zpp_1000_%d' % t, 'zpp_0001_%d' % t, 'zpp_0100_%d' % t, 'zpp_0010_%d' % t)
            if not m2:
                return m2
            t0 = tripleRatioCondition('z_0100_%d' % t, 'zp_0010_%d' % t, 'zpp_0001_%d' % t)
            if not t0:
                return t0
            t1 = tripleRatioCondition('z_1000_%d' % t, 'zp_0001_%d' % t, 'zpp_0010_%d' % t)
            if not t1:
                return t1
            t2 = tripleRatioCondition('z_0001_%d' % t, 'zp_1000_%d' % t, 'zpp_0100_%d' % t)
            if not t2:
                return t2
            t3 = tripleRatioCondition('z_0010_%d' % t, 'zp_0100_%d' % t, 'zpp_1000_%d' % t)
            if not t3:
                return t3
        return True

    def is_geometric(self, epsilon=1e-06):
        """
        Returns true if all shapes corresponding to this solution have positive
        imaginary part.

        If the solutions are exact, it returns true if one of the corresponding
        numerical solutions is geometric.

        An optional epsilon can be given. An imaginary part of a shape is
        considered positive if it is larger than this epsilon.
        """
        if self._is_numerical:
            for v in self.values():
                if not v.imag() > 0:
                    return False
            return True
        else:
            for numerical_sol in self.numerical():
                if numerical_sol.is_geometric(epsilon):
                    return True
            return False