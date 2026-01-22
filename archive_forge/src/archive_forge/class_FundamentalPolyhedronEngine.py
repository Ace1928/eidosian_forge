from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
class FundamentalPolyhedronEngine(McomplexEngine):

    @staticmethod
    def from_manifold_and_shapes(manifold, shapes, normalize_matrices=False, match_kernel=True):
        """
        Given a SnapPy.Manifold and shapes (which can be numbers or intervals),
        create a t3mlite.Mcomplex for the fundamental polyhedron that the
        SnapPea kernel computed, assign each vertex of it to a point on the
        boundary of upper half space H^3, and compute the matrices pairing the
        faces of the fundamental polyhedron. The matrices will have determinant
        one if normalize_matrices is True.

        Some notes about the vertices: We use the one-point
        compactification to represent the boundary of H^3, i.e., we
        either assign a complex number (or interval) to a vertex or
        Infinity (a sentinel in kernel_structures). If match_kernel is True,
        the vertices are at the same positions than in the SnapPea kernel.
        This has the disadvantage that the matrices computed that way no longer
        have entries in the trace field. Use match_kernel is False for matrices
        over the trace field (e.g., obtain the quaternion algebra).

        Some notes about the matrices: If normalize_matrices is False, the
        product of a matrix for a generator and its inverse is not necessarily
        the identity, but a multiple of the identity.
        Even if normalize_matrices is True, the product of matrices
        corresponding to the letters in a relation might still yield minus the
        identity (i.e., we do not lift to SL(2,C)).

        >>> M = Manifold("m004")
        >>> F = FundamentalPolyhedronEngine.from_manifold_and_shapes(
        ...      M, M.tetrahedra_shapes('rect'))

        The above code adds the given shapes to each edge (here 01) of each
        tetrahedron::

            >>> from snappy.snap.t3mlite import simplex
            >>> F.mcomplex.Tetrahedra[0].ShapeParameters[simplex.E01] # doctest: +NUMERIC6
            0.500000000000000 + 0.866025403784438*I

        And annotates each face (here 1) of each tetrahedron with the
        corresponding generator (here, the inverse of the second generator)
        or 0 if the face is internal to the fundamental polyhedron::

            >>> F.mcomplex.Tetrahedra[0].GeneratorsInfo[simplex.F1]
            -2

        This information is also available in a dict keyed by generator.
        For each generator, it gives a list of the corresponding face pairing
        data (there might be multiple face pairings corresponding to the same
        generator). The face pairing data consists of a pair of t3mlite.Corner's
        indicating the paired faces as well as the permutation to take one
        face to the other.
        Here, for example, the generator corresponds to exactly one face
        pairing of face 2 of tet 1 to face 1 of tet0 such that face 2 is
        taken to face 1 by the permutation (3, 0, 1, 2)::

            >>> F.mcomplex.Generators[2]
            [((<F2 of tet1>, <F1 of tet0>), (3, 0, 1, 2))]

        The four vertices of tetrahedron 1::

            >>> for v in simplex.ZeroSubsimplices: # doctest: +NUMERIC6
            ...     F.mcomplex.Tetrahedra[1].Class[v].IdealPoint
            'Infinity'
            0.000000000000000
            0.866025403784439 - 0.500000000000000*I
            0.866025403784439 + 0.500000000000000*I

        The matrix for generator 1 (of the unsimplified presentation)::

            >>> F.mcomplex.GeneratorMatrices[1] # doctest: +NUMERIC6 +ELLIPSIS
            [   -0.577350269189626 - 1.00000000000000*I    0.500000000000000 + 0.288675134594813*I...]
            [  -0.500000000000000 - 0.288675134594813*I 0.577350269189626 + 2.22044604925031e-16*I...]

        Get the cusp that a vertex of the fundamental polyhedron corresponds
        to::

            >>> F.mcomplex.Tetrahedra[1].Class[simplex.V0].SubsimplexIndexInManifold
            0

        """
        m = t3m.Mcomplex(manifold)
        f = FundamentalPolyhedronEngine(m)
        t = TransferKernelStructuresEngine(m, manifold)
        t.add_shapes(shapes)
        t.choose_and_transfer_generators(compute_corners=True, centroid_at_origin=False)
        f.unglue()
        if match_kernel:
            init_verts = f.init_vertices_kernel()
        else:
            init_verts = f.init_vertices()
        f.visit_tetrahedra_to_compute_vertices(m.ChooseGenInitialTet, init_verts)
        f.compute_matrices(normalize_matrices=normalize_matrices)
        return f

    def unglue(self):
        """
        It will unglue all face-pairings corresponding to generators.
        What is left is a fundamental polyhedron.

        It assumes that GeneratorsInfo has been set (by the
        TranferKernelStructuresEngine).

        Besides ungluing, it will add the field Generators to the Mcomplex
        and SubsimplexIndexInManifold to each Vertex, Edge, Face, see
        examples in from_manifold_and_shapes.
        """
        originalSubsimplexIndices = [[tet.Class[subsimplex].Index for subsimplex in range(1, 15)] for tet in self.mcomplex.Tetrahedra]
        self.mcomplex.Generators = {}
        for tet in self.mcomplex.Tetrahedra:
            for face in simplex.TwoSubsimplices:
                g = tet.GeneratorsInfo[face]
                if g != 0:
                    l = self.mcomplex.Generators.setdefault(g, [])
                    l.append(((Corner(tet, face), Corner(tet.Neighbor[face], tet.Gluing[face].image(face))), tet.Gluing[face]))
        for g, pairings in self.mcomplex.Generators.items():
            if g > 0:
                for corners, perm in pairings:
                    for corner in corners:
                        corner.Tetrahedron.attach(corner.Subsimplex, None, None)
        self.mcomplex.rebuild()
        for tet, o in zip(self.mcomplex.Tetrahedra, originalSubsimplexIndices):
            for subsimplex, index in enumerate(o):
                tet.Class[subsimplex + 1].SubsimplexIndexInManifold = index

    def visit_tetrahedra_to_compute_vertices(self, init_tet, init_vertices):
        """
        Computes the positions of the vertices of fundamental polyhedron in
        the boundary of H^3, assuming the Mcomplex has been unglued and
        ShapeParameters were assigned to the tetrahedra.

        It starts by assigning the vertices of the given init_tet using
        init_vertices.
        """
        for vertex in self.mcomplex.Vertices:
            vertex.IdealPoint = None
        for tet in self.mcomplex.Tetrahedra:
            tet.visited = False
        self.mcomplex.InitialTet = init_tet
        for v, idealPoint in init_vertices.items():
            init_tet.Class[v].IdealPoint = idealPoint
        init_tet.visited = True
        queue = [init_tet]
        while len(queue) > 0:
            tet = queue.pop(0)
            for F in simplex.TwoSubsimplices:
                if bool(tet.Neighbor[F]) != bool(tet.GeneratorsInfo[F] == 0):
                    raise Exception('Improper fundamental domain, probably a bug in unglue code')
                S = tet.Neighbor[F]
                if S and (not S.visited):
                    perm = tet.Gluing[F]
                    for V in _VerticesInFace[F]:
                        vertex_class = S.Class[perm.image(V)]
                        if vertex_class.IdealPoint is None:
                            vertex_class.IdealPoint = tet.Class[V].IdealPoint
                    _compute_fourth_corner(S)
                    S.visited = True
                    queue.append(S)

    def init_vertices(self):
        """
        Computes vertices for the initial tetrahedron such that vertex 0, 1
        and 2 are at Infinity, 0 and z.
        """
        tet = self.mcomplex.ChooseGenInitialTet
        z = tet.ShapeParameters[simplex.E01]
        CF = z.parent()
        return {simplex.V0: Infinity, simplex.V1: CF(0), simplex.V2: CF(1), simplex.V3: z}

    def init_vertices_kernel(self):
        """
        Computes vertices for the initial tetrahedron matching the choices
        made by the SnapPea kernel.
        """
        tet = self.mcomplex.ChooseGenInitialTet
        candidates = []
        for perm in Perm4.A4():
            z = tet.ShapeParameters[perm.image(simplex.E01)]
            CF = z.parent()
            sqrt_z = z.sqrt()
            sqrt_z_inv = CF(1) / sqrt_z
            candidate = {perm.image(simplex.V0): Infinity, perm.image(simplex.V1): CF(0), perm.image(simplex.V2): sqrt_z_inv, perm.image(simplex.V3): sqrt_z}
            if _are_vertices_close_to_kernel(candidate, tet.SnapPeaIdealVertices):
                candidates.append(candidate)
        if len(candidates) == 1:
            return candidates[0]
        raise Exception('Could not match vertices to vertices from SnapPea kernel')

    def compute_matrices(self, normalize_matrices=False):
        """
        Assuming positions were assigned to the vertices, adds
        GeneratorMatrices to the Mcomplex which assigns a matrix to each
        generator.

        Compute generator matrices:

        >>> M = Manifold("s776")
        >>> F = FundamentalPolyhedronEngine.from_manifold_and_shapes(
        ...      M, M.tetrahedra_shapes('rect'), normalize_matrices = True)
        >>> generatorMatrices = F.mcomplex.GeneratorMatrices

        Given a letter such as 'a' or 'A', return matrix for corresponding
        generator:

        >>> def letterToMatrix(l, generatorMatrices):
        ...     g = ord(l.lower()) - ord('a') + 1
        ...     if l.isupper():
        ...         g = -g
        ...     return generatorMatrices[g]

        Check that relations are fulfilled up to sign:

        >>> def p(L): return reduce(lambda x, y: x * y, L)
        >>> def close_to_identity(m, epsilon = 1e-12):
        ...     return abs(m[(0,0)] - 1) < epsilon and abs(m[(1,1)] - 1) < epsilon and abs(m[(0,1)]) < epsilon and abs(m[(1,0)]) < epsilon
        >>> def close_to_pm_identity(m, epsilon = 1e-12):
        ...     return close_to_identity(m, epsilon) or close_to_identity(-m, epsilon)
        >>> G = M.fundamental_group(simplify_presentation = False)
        >>> for rel in G.relators():
        ...     close_to_pm_identity(p([letterToMatrix(l, generatorMatrices) for l in rel]))
        True
        True
        True
        True

        """
        z = self.mcomplex.Tetrahedra[0].ShapeParameters[simplex.E01]
        CF = z.parent()
        self.mcomplex.GeneratorMatrices = {0: matrix([[CF(1), CF(0)], [CF(0), CF(1)]])}
        for g, pairings in self.mcomplex.Generators.items():
            if g > 0:
                m = _compute_pairing_matrix(pairings[0])
                if normalize_matrices:
                    m = m / m.det().sqrt()
                self.mcomplex.GeneratorMatrices[g] = m
                self.mcomplex.GeneratorMatrices[-g] = _adjoint2(m)

    def matrices_for_presentation(self, G, match_kernel=False):
        """
        Given the result of M.fundamental_group(...) where M is the
        corresponding SnapPy.Manifold, return the matrices for that
        presentation of the fundamental polyhedron.

        The GeneratorMatrices computed here are for the face-pairing
        presentation with respect to the fundamental polyhedron.
        That presentation can be simplfied by M.fundamental_group(...)
        and this function will compute the matrices for the simplified
        presentation from the GeneratorMatrices.

        If match_kernel is True, it will flip the signs of some of
        the matrices to match the ones in the given G (which were determined
        by the SnapPea kernel).

        This makes the result stable when changing precision (when normalizing
        matrices with determinant -1, sqrt(-1) might jump between i and -i when
        increasing precision).
        """
        num_generators = len(self.mcomplex.GeneratorMatrices) // 2
        matrices = [self.mcomplex.GeneratorMatrices[g + 1] for g in range(num_generators)]
        result = _perform_word_moves(matrices, G)
        if match_kernel:
            return _negate_matrices_to_match_kernel(result, G)
        else:
            return result