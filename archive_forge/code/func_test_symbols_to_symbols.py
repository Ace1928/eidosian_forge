import pytest
from ase import Atoms
from ase.build import molecule
from ase.symbols import Symbols
def test_symbols_to_symbols(symbols):
    assert all(Symbols(symbols.numbers) == symbols)