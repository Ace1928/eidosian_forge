import os
import sys
import shutil
import time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import jsonio
from ase.io.ulm import open as ulmopen
from ase.parallel import paropen, world, barrier
from ase.calculators.singlepoint import (SinglePointCalculator,
def _write_atoms(self, atoms):
    self._call_observers(self.pre_observers)
    self.log('Beginning to write frame ' + str(self.nframes))
    framedir = self._make_framedir(self.nframes)
    datatypes = {}
    for k, v in self.datatypes.items():
        if v == 'once':
            v = self.nframes == 0
        datatypes[k] = v
    smalldata = {'pbc': atoms.get_pbc(), 'cell': atoms.get_cell(), 'natoms': atoms.get_global_number_of_atoms(), 'constraints': atoms.constraints}
    if datatypes.get('energy'):
        try:
            smalldata['energy'] = atoms.get_potential_energy()
        except (RuntimeError, PropertyNotImplementedError):
            self.datatypes['energy'] = False
    if datatypes.get('stress'):
        try:
            smalldata['stress'] = atoms.get_stress()
        except PropertyNotImplementedError:
            self.datatypes['stress'] = False
    self.backend.write_small(framedir, smalldata)
    if datatypes.get('positions'):
        self.backend.write(framedir, 'positions', atoms.get_positions())
    if datatypes.get('numbers'):
        self.backend.write(framedir, 'numbers', atoms.get_atomic_numbers())
    if datatypes.get('tags'):
        if atoms.has('tags'):
            self.backend.write(framedir, 'tags', atoms.get_tags())
        else:
            self.datatypes['tags'] = False
    if datatypes.get('masses'):
        if atoms.has('masses'):
            self.backend.write(framedir, 'masses', atoms.get_masses())
        else:
            self.datatypes['masses'] = False
    if datatypes.get('momenta'):
        if atoms.has('momenta'):
            self.backend.write(framedir, 'momenta', atoms.get_momenta())
        else:
            self.datatypes['momenta'] = False
    if datatypes.get('magmoms'):
        if atoms.has('initial_magmoms'):
            self.backend.write(framedir, 'magmoms', atoms.get_initial_magnetic_moments())
        else:
            self.datatypes['magmoms'] = False
    if datatypes.get('forces'):
        try:
            x = atoms.get_forces()
        except (RuntimeError, PropertyNotImplementedError):
            self.datatypes['forces'] = False
        else:
            self.backend.write(framedir, 'forces', x)
            del x
    if datatypes.get('energies'):
        try:
            x = atoms.get_potential_energies()
        except (RuntimeError, PropertyNotImplementedError):
            self.datatypes['energies'] = False
        else:
            self.backend.write(framedir, 'energies', x)
            del x
    for label, source, once in self.extra_data:
        if self.nframes == 0 or not once:
            if source is not None:
                x = source()
            else:
                x = atoms.get_array(label)
            self.backend.write(framedir, label, x)
            del x
            if once:
                self.datatypes[label] = 'once'
            else:
                self.datatypes[label] = True
    if self.nframes == 0:
        metadata = {'datatypes': self.datatypes}
        self._write_metadata(metadata)
    self._write_nframes(self.nframes + 1)
    self._call_observers(self.post_observers)
    self.log('Done writing frame ' + str(self.nframes))
    self.nframes += 1