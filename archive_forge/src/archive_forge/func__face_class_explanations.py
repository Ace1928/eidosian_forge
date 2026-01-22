from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def _face_class_explanations(self):
    """
        A list giving for all the face classes a string s_face_tetrahedron.
        """

    def process_triangle(triangle):
        embedding = triangle.getEmbedding(0)
        face = embedding.getFace()
        tet = self.tetrahedronIndex(embedding.getTetrahedron())
        return 's_%d_%d' % (face, tet)
    return [process_triangle(triangle) for triangle in self.getTriangles()]