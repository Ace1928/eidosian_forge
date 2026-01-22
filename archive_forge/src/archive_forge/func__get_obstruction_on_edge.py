from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def _get_obstruction_on_edge(self, obstruction_class, tet, v0, v1):
    """
        Reimplements _get_obstruction_on_edge from
        addl_code/ptolemy_equations.c
        """
    if v0 > v1:
        return -self._get_obstruction_on_edge(obstruction_class, tet, v1, v0)
    if v1 != v0 + 1:
        return 0
    s = [NTriangulationForPtolemy._sign_of_tetrahedron_face(tet, i) * obstruction_class[self._index_of_tetrahedron_face(tet, i)] for i in range(4)]
    if v0 == 0:
        return -s[0] - s[1] - s[3]
    if v0 == 1:
        return s[0] + s[1]
    if v0 == 2:
        return -s[1]
    raise Exception('Should not get here')