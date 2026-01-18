import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
@pytest.mark.parametrize('setting_name, ref_setting', [('hexagonal', 1), ('trigonal', 2), ('rhombohedral', 2)])
def test_spacegroup_named_setting(setting_name, ref_setting):
    """The rhombohedral crystal system signifies setting=2"""
    ciffile = io.BytesIO("data_test\n_space_group_crystal_system {}\n_symmetry_space_group_name_H-M         'R-3m'\n".format(setting_name).encode('ascii'))
    blocks = list(parse_cif(ciffile))
    assert len(blocks) == 1
    spg = blocks[0].get_spacegroup(False)
    assert int(spg) == 166
    assert spg.setting == ref_setting