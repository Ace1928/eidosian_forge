import os
import pytest
import numpy as np
import ase
import ase.lattice.cubic
from ase.calculators.castep import (Castep, CastepOption,
def test_set_kpoints(testing_calculator):
    c = testing_calculator
    c.set_kpts([(0.0, 0.0, 0.0, 1.0)])
    assert c.cell.kpoint_list.value == '0.0 0.0 0.0 1.0'
    c.set_kpts(((0.0, 0.0, 0.0, 0.25), (0.25, 0.25, 0.3, 0.75)))
    assert c.cell.kpoint_list.value == '0.0 0.0 0.0 0.25\n0.25 0.25 0.3 0.75'
    c.set_kpts(c.cell.kpoint_list.value.split('\n'))
    assert c.cell.kpoint_list.value == '0.0 0.0 0.0 0.25\n0.25 0.25 0.3 0.75'
    c.set_kpts([3, 3, 2])
    assert c.cell.kpoint_mp_grid.value == '3 3 2'
    c.set_kpts(None)
    assert c.cell.kpoints_list.value is None
    assert c.cell.kpoint_list.value is None
    assert c.cell.kpoint_mp_grid.value is None
    c.set_kpts('2 2 3')
    assert c.cell.kpoint_mp_grid.value == '2 2 3'
    c.set_kpts({'even': True, 'gamma': True})
    assert c.cell.kpoint_mp_grid.value == '2 2 2'
    assert c.cell.kpoint_mp_offset.value == '0.25 0.25 0.25'
    c.set_kpts({'size': (2, 2, 4), 'even': False})
    assert c.cell.kpoint_mp_grid.value == '3 3 5'
    assert c.cell.kpoint_mp_offset.value == '0.0 0.0 0.0'
    atoms = ase.build.bulk('Ag')
    atoms.calc = c
    c.set_kpts({'density': 10, 'gamma': False, 'even': None})
    assert c.cell.kpoint_mp_grid.value == '27 27 27'
    assert c.cell.kpoint_mp_offset.value == '0.018519 0.018519 0.018519'
    c.set_kpts({'spacing': 1 / (np.pi * 10), 'gamma': False, 'even': True})
    assert c.cell.kpoint_mp_grid.value == '28 28 28'
    assert c.cell.kpoint_mp_offset.value == '0.0 0.0 0.0'