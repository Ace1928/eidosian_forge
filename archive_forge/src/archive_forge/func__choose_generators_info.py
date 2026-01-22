from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def _choose_generators_info(self):
    """
        Provides information about the choices mades when regina computed the
        simplified fundamental group in a structure similar to SnapPy.
        """
    if not hasattr(self, 'maximalForestInDualSkeleton'):
        raise Exception('Version of regina does not have python wrapping for maximalForestInDualSkeleton')
    non_generator_triangles = self.maximalForestInDualSkeleton()
    generator_triangles = [triangle for triangle in self.getTriangles() if triangle not in non_generator_triangles]

    def get_neighbors(tet):
        """
            Given a tetrahedron, return the indices of the 4 neighboring
            tetrahedra
            """
        return [self.tetrahedronIndex(tet.getAdjacentTetrahedron(face)) for face in range(4)]

    def get_gluings(tet):
        """
            Given a tetrahedron, return the four face gluings.
            """
        return [tet.getAdjacentTetrahedronGluing(face) for face in range(4)]

    def get_generator(tet, face):
        """
            Given a tetrahedron and a face (0, ..., 3), return whether it
            corresponds to an inbound or outbound generator.
            """
        triangle = tet.getTriangle(face)
        if triangle not in generator_triangles:
            return 0
        gen = generator_triangles.index(triangle) + 1
        canonical_embed = triangle.getEmbedding(0)
        if canonical_embed.getTetrahedron() == tet and canonical_embed.getFace() == face:
            return gen
        else:
            return -gen

    def get_generators(tet):
        """
            Given a tetrahedron, return for each face which inbound
            or outbound generator it belongs to.
            """
        return [get_generator(tet, face) for face in range(4)]
    return [{'index': index, 'neighbors': get_neighbors(tet), 'gluings': get_gluings(tet), 'generators': get_generators(tet)} for index, tet in enumerate(self.getTetrahedra())]