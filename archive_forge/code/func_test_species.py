import pytest
from ase import Atoms
from ase.build import molecule
from ase.symbols import Symbols
def test_species(atoms):
    assert atoms.symbols.species() == set(atoms.symbols)