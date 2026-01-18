import pytest
from ase.utils import xwopen
def test_xwopen():
    with xwopen(filename) as fd:
        fd.write(poem)
    assert fd.closed
    with open(filename, 'rb') as fd:
        assert fd.read() == poem