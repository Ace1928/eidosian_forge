import numpy as np
from ..Qt import QtGui
def vertexes(self, indexed=None):
    """Return an array (N,3) of the positions of vertexes in the mesh. 
        By default, each unique vertex appears only once in the array.
        If indexed is 'faces', then the array will instead contain three vertexes
        per face in the mesh (and a single vertex may appear more than once in the array)."""
    if indexed is None:
        if self._vertexes is None and self._vertexesIndexedByFaces is not None:
            self._computeUnindexedVertexes()
        return self._vertexes
    elif indexed == 'faces':
        if self._vertexesIndexedByFaces is None and self._vertexes is not None:
            self._vertexesIndexedByFaces = self._vertexes[self.faces()]
        return self._vertexesIndexedByFaces
    else:
        raise Exception("Invalid indexing mode. Accepts: None, 'faces'")