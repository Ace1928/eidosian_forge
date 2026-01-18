import numpy as np
import pytest
from ase.io.formats import ioformats, match_magic
@pytest.mark.parametrize('header', lammpsdump_headers())
def test_recognize_lammpsdump(header):
    fmt_name = 'lammps-dump-text'
    fmt = match_magic(header.encode('ascii'))
    assert fmt.name == fmt_name