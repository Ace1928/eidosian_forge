import os
import warnings
import numpy as np
import ase
from ase.data import atomic_masses
from ase.geometry import cellpar_to_cell
import collections
from functools import reduce
def write_arrays(self, atoms, frame, arrays):
    self._open()
    self._call_observers(self.pre_observers)
    for array in arrays:
        data = atoms.get_array(array)
        if array in self.extra_per_file_vars:
            if np.any(self._get_variable(array) != data):
                raise ValueError('Trying to write Atoms object with incompatible data for the {0} array.'.format(array))
        else:
            self._add_array(atoms, array, data.dtype, data.shape)
            self._get_variable(array)[frame] = data
    self._call_observers(self.post_observers)
    self._close()