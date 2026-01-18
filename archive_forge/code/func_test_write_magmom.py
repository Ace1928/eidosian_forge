import pytest
from unittest import mock
import numpy as np
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.vasp.create_input import _args_without_comment
from ase.calculators.vasp.create_input import _to_vasp_bool, _from_vasp_bool
from ase.build import bulk
@pytest.mark.parametrize('list_func', [list, tuple, np.array])
def test_write_magmom(magmoms_factory, list_func, nacl, vaspinput_factory, assert_magmom_equal_to_incar_value, testdir):
    """Test writing magnetic moments to INCAR, and ensure we can do it
    passing different types of sequences"""
    magmom = magmoms_factory(nacl)
    vaspinput = vaspinput_factory(atoms=nacl, magmom=magmom, ispin=2)
    assert vaspinput.spinpol
    assert_magmom_equal_to_incar_value(nacl, magmom, vaspinput)