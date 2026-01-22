import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
class VertexVectorField(VertexBase):
    """
    Add homology properties of a scalar field f: R^n --> R^m associated with
    the geometry built from the VertexBase class.
    """

    def __init__(self, x, sfield=None, vfield=None, field_args=(), vfield_args=(), g_cons=None, g_cons_args=(), nn=None, index=None):
        super().__init__(x, nn=nn, index=index)
        raise NotImplementedError('This class is still a work in progress')