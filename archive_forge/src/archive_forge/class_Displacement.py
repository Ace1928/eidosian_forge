from math import pi, sqrt, log
import sys
import numpy as np
from pathlib import Path
import ase.units as units
import ase.io
from ase.parallel import world, paropen
from ase.utils.filecache import get_json_cache
from .data import VibrationsData
from collections import namedtuple
class Displacement(namedtuple('Displacement', ['a', 'i', 'sign', 'ndisp', 'vib'])):

    @property
    def name(self):
        if self.sign == 0:
            return 'eq'
        axisname = 'xyz'[self.i]
        dispname = self.ndisp * ' +-'[self.sign]
        return f'{self.a}{axisname}{dispname}'

    @property
    def _cached(self):
        return self.vib.cache[self.name]

    def forces(self):
        return self._cached['forces'].copy()

    @property
    def step(self):
        return self.ndisp * self.sign * self.vib.delta

    def dipole(self):
        return self._cached['dipole'].copy()

    def save_ov_nn(self, ov_nn):
        np.save(self.name + '.ov', ov_nn)

    def load_ov_nn(self):
        return np.load(self.name + '.ov.npy')

    @property
    def _exname(self):
        return Path(self.vib.exname) / f'ex.{self.name}{self.vib.exext}'

    def calculate_and_save_static_polarizability(self, atoms):
        exobj = self.vib._new_exobj()
        excitation_data = exobj.calculate(atoms)
        np.savetxt(self._exname, excitation_data)

    def load_static_polarizability(self):
        return np.loadtxt(self._exname)

    def read_exobj(self):
        return self.vib.read_exobj(str(self._exname))

    def calculate_and_save_exlist(self, atoms):
        excalc = self.vib._new_exobj()
        exlist = excalc.calculate(atoms)
        exlist.write(str(self._exname))