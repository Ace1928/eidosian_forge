import os
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from ase import units, Atoms
import ase.io
from ase.calculators.qmmm import ForceConstantCalculator
from ase.vibrations import Vibrations, VibrationsData
from ase.thermochemistry import IdealGasThermo
def test_tabulate_energies(self):
    energies = np.array([1.0, complex(2.0, 1.0), complex(1.0, 0.001)])
    table = VibrationsData._tabulate_from_energies(energies, im_tol=0.01)
    for sep_row in (0, 2, 6):
        assert table[sep_row] == '-' * 21
    assert tuple(table[1].strip().split()) == ('#', 'meV', 'cm^-1')
    expected_rows = [('0', '1000.0', '8065.5'), ('1', '1000.0i', '8065.5i'), ('2', '1000.0', '8065.5')]
    for row, expected in zip(table[3:6], expected_rows):
        assert tuple(row.split()) == expected
    assert table[7].split()[2] == '2.000'
    assert len(table) == 8