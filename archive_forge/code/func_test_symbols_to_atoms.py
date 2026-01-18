import pytest
from ase import Atoms
from ase.build import molecule
from ase.symbols import Symbols
def test_symbols_to_atoms(symbols):
    assert all(Atoms(symbols).symbols == symbols)