from the :meth:`.cifti2.Cifti2Header.get_axis` method on the header object
import abc
from operator import xor
import numpy as np
from . import cifti2
@property
def volume_mask(self):
    """
        (N, ) boolean array which is true for any element on the surface
        """
    return np.vectorize(lambda name: name not in self.nvertices.keys())(self.name)