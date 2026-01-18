import pytest
from ase import Atoms
from ase.build import molecule
from ase.symbols import Symbols
def test_search_two(atoms):
    indices = atoms.symbols.search('CO')
    assert all((sym in {'C', 'O'} for sym in atoms.symbols[indices]))