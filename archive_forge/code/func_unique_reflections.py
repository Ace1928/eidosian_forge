import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def unique_reflections(self, hkl):
    """Returns a subset *hkl* containing only the symmetry-unique
        reflections.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sg.unique_reflections([[ 2,  0,  0],
        ...                        [ 0, -2,  0],
        ...                        [ 2,  2,  0],
        ...                        [ 0, -2, -2]])
        array([[2, 0, 0],
               [2, 2, 0]])
        """
    hkl = np.array(hkl, dtype=int, ndmin=2)
    hklnorm = self.symmetry_normalised_reflections(hkl)
    perm = np.lexsort(hklnorm.T)
    iperm = perm.argsort()
    xmask = np.abs(np.diff(hklnorm[perm], axis=0)).any(axis=1)
    mask = np.concatenate(([True], xmask))
    imask = mask[iperm]
    return hkl[imask]