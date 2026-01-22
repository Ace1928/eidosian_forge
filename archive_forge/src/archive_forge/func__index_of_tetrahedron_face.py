from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def _index_of_tetrahedron_face(self, tetrahedron, face):
    """
        Helper for computing the homology. We call a generator
        of the simplicial chain C_2(M, \\partial M) a face class.
        There are 2 * #tetrahedra faces classes and we need to
        index them and find a representative for each.
        A representative is a pair of (tetrahedron, face).

        This method returns the index of the face class given
        a representative.
        regina already gives indices to the triangles, so we can
        use that here.

        In addl_code/ptolemy_equations.c, this was done in
        _fill_tet_face_to_index_data .
        """
    return self.triangleIndex(tetrahedron.getTriangle(face))