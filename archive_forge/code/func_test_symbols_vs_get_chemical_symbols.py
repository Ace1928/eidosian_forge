import pytest
from ase import Atoms
from ase.build import molecule
from ase.symbols import Symbols
def test_symbols_vs_get_chemical_symbols(atoms):
    assert atoms.get_chemical_symbols() == list(atoms.symbols)