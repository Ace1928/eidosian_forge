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
def test_todict(self, n2_data, n2_vibdata):
    vib_data_dict = n2_vibdata.todict()
    assert vib_data_dict['indices'] is None
    assert_array_almost_equal(vib_data_dict['atoms'].positions, n2_data['atoms'].positions)
    assert_array_almost_equal(vib_data_dict['hessian'], n2_data['hessian'])