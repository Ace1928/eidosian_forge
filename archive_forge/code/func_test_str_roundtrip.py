import pytest
from ase import Atoms
from ase.build import molecule
from ase.symbols import Symbols
def test_str_roundtrip(symbols):
    string = str(symbols)
    newsymbols = Symbols.fromsymbols(string)
    assert (symbols == newsymbols).all()