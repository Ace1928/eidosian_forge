import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
class VertexCacheIndex(VertexCacheBase):

    def __init__(self):
        """
        Class for a vertex cache for a simplicial complex without an associated
        field. Useful only for building and visualising a domain complex.

        Parameters
        ----------
        """
        super().__init__()
        self.Vertex = VertexCube

    def __getitem__(self, x, nn=None):
        try:
            return self.cache[x]
        except KeyError:
            self.index += 1
            xval = self.Vertex(x, index=self.index)
            self.cache[x] = xval
            return self.cache[x]