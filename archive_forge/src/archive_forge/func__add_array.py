import os
import warnings
import numpy as np
import ase
from ase.data import atomic_masses
from ase.geometry import cellpar_to_cell
import collections
from functools import reduce
def _add_array(self, atoms, array_name, type, shape):
    if not self._has_variable(array_name):
        dims = [self._frame_dim]
        for i in shape:
            if i == len(atoms):
                dims += [self._atom_dim]
            elif i == 3:
                dims += [self._spatial_dim]
            elif i == 6:
                if self._Voigt_dim not in self.nc.dimensions:
                    self.nc.createDimension(self._Voigt_dim, 6)
                dims += [self._Voigt_dim]
            else:
                raise TypeError("Don't know how to dump array of shape {0} into NetCDF trajectory.".format(shape))
        if hasattr(type, 'char'):
            t = self.dtype_conv.get(type.char, type)
        else:
            t = type
        self.nc.createVariable(array_name, t, dims)