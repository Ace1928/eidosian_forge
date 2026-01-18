from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import UnivariateSpline
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
def zval_dict_from_potcar(potcar):
    """
    Creates zval_dictionary for calculating the ionic polarization from
    Potcar object.

    potcar: Potcar object
    """
    zval_dict = {}
    for p in potcar:
        zval_dict.update({p.element: p.ZVAL})
    return zval_dict