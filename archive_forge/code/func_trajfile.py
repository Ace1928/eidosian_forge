import pytest
from ase import Atom, Atoms
from ase.io import Trajectory, read
from ase.constraints import FixBondLength
from ase.calculators.calculator import PropertyNotImplementedError
@pytest.fixture
def trajfile(trajfile_and_images):
    return trajfile_and_images[0]