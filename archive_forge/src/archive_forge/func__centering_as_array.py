import copy
import numbers
from math import cos, sin, pi
import numpy as np
import ase.units as units
from ase.atom import Atom
from ase.cell import Cell
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.data import atomic_masses, atomic_masses_common
from ase.geometry import (wrap_positions, find_mic, get_angles, get_distances,
from ase.symbols import Symbols, symbols2numbers
from ase.utils import deprecated
def _centering_as_array(self, center):
    if isinstance(center, str):
        if center.lower() == 'com':
            center = self.get_center_of_mass()
        elif center.lower() == 'cop':
            center = self.get_positions().mean(axis=0)
        elif center.lower() == 'cou':
            center = self.get_cell().sum(axis=0) / 2
        else:
            raise ValueError('Cannot interpret center')
    else:
        center = np.array(center, float)
    return center