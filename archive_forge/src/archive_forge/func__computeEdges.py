import numpy as np
from ..Qt import QtGui
def _computeEdges(self):
    if not self.hasFaceIndexedData():
        nf = len(self._faces)
        edges = np.empty(nf * 3, dtype=[('i', np.uint, 2)])
        edges['i'][0:nf] = self._faces[:, :2]
        edges['i'][nf:2 * nf] = self._faces[:, 1:3]
        edges['i'][-nf:, 0] = self._faces[:, 2]
        edges['i'][-nf:, 1] = self._faces[:, 0]
        mask = edges['i'][:, 0] > edges['i'][:, 1]
        edges['i'][mask] = edges['i'][mask][:, ::-1]
        self._edges = np.unique(edges)['i']
    elif self._vertexesIndexedByFaces is not None:
        verts = self._vertexesIndexedByFaces
        edges = np.empty((verts.shape[0], 3, 2), dtype=np.uint)
        nf = verts.shape[0]
        edges[:, 0, 0] = np.arange(nf) * 3
        edges[:, 0, 1] = edges[:, 0, 0] + 1
        edges[:, 1, 0] = edges[:, 0, 1]
        edges[:, 1, 1] = edges[:, 1, 0] + 1
        edges[:, 2, 0] = edges[:, 1, 1]
        edges[:, 2, 1] = edges[:, 0, 0]
        self._edges = edges
    else:
        raise Exception('MeshData cannot generate edges--no faces in this data.')