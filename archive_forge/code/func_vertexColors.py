import numpy as np
from ..Qt import QtGui
def vertexColors(self, indexed=None):
    """
        Return an array (Nv, 4) of vertex colors.
        If indexed=='faces', then instead return an indexed array
        (Nf, 3, 4). 
        """
    if indexed is None:
        return self._vertexColors
    elif indexed == 'faces':
        if self._vertexColorsIndexedByFaces is None:
            self._vertexColorsIndexedByFaces = self._vertexColors[self.faces()]
        return self._vertexColorsIndexedByFaces
    else:
        raise Exception("Invalid indexing mode. Accepts: None, 'faces'")