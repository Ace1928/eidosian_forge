import pytest
import numpy as np
from ase.io.zmatrix import parse_zmatrix
@pytest.mark.parametrize('zmat, defs', tests)
def test_zmatrix_diborane(zmat, defs):
    assert parse_zmatrix(zmat, defs=defs).positions == pos_ref