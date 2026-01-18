import os
import numpy as np
import numpy.testing
import unittest
import ase
import ase.build
import ase.io
from ase.io.vasp import write_vasp_xdatcar
from ase.calculators.calculator import compare_atoms
def test_roundtrip_single_atoms(self):
    atoms = ase.build.bulk('Ge')
    ase.io.write(self.outfile, atoms, format='vasp-xdatcar')
    roundtrip_atoms = ase.io.read(self.outfile)
    self.assert_atoms_almost_equal(atoms, roundtrip_atoms)