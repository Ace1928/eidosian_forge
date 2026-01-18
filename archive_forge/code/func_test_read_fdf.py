from io import StringIO
from pathlib import Path
import numpy as np
import pytest
from ase.io import read
from ase.io.siesta import read_struct_out, read_fdf
from ase.units import Bohr
def test_read_fdf():
    dct = read_fdf(StringIO(sample_fdf))
    ref = dict(potatoes=['5'], coffee=['6.5'], spam=[['1', '2.5', 'hello']])
    assert dct == ref