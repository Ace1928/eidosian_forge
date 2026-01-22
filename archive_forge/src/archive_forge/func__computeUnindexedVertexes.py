import numpy as np
from ..Qt import QtGui
def _computeUnindexedVertexes(self):
    faces = self._vertexesIndexedByFaces
    verts = {}
    self._faces = np.empty(faces.shape[:2], dtype=np.uint)
    self._vertexes = []
    self._vertexFaces = []
    self._faceNormals = None
    self._vertexNormals = None
    for i in range(faces.shape[0]):
        face = faces[i]
        for j in range(face.shape[0]):
            pt = face[j]
            pt2 = tuple([round(x * 100000000000000.0) for x in pt])
            index = verts.get(pt2, None)
            if index is None:
                self._vertexes.append(pt)
                self._vertexFaces.append([])
                index = len(self._vertexes) - 1
                verts[pt2] = index
            self._vertexFaces[index].append(i)
            self._faces[i, j] = index
    self._vertexes = np.array(self._vertexes, dtype=np.float32)