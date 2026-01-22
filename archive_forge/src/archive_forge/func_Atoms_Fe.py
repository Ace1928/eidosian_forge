import pytest
import numpy as np
from ase import Atom
from ase.build import bulk
import ase.io
from ase import units
from ase.md.verlet import VelocityVerlet
@pytest.fixture
def Atoms_Fe(lammps_data_file_Fe):
    Atoms_Fe = ase.io.read(lammps_data_file_Fe, format='lammps-data', Z_of_type={1: 26}, units='real')
    return Atoms_Fe