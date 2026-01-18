import numpy as np
import pytest
from ase.lattice import MCLC
def test_zero_npoints(lat):
    path = lat.bandpath(npoints=0)
    assert path.path == lat.special_path
    assert len(path.kpts) == len(path.get_linear_kpoint_axis()[2])