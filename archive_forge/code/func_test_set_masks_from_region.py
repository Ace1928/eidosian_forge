import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
def test_set_masks_from_region(at0, qm_calc, mm_calc):
    """
    Test setting masks from region array
    """
    qmmm = at0.calc
    region = qmmm.get_region_from_masks(at0)
    r = at0.get_distances(0, np.arange(len(at0)), mic=True)
    R_QM = 0.001
    qm_mask = r < R_QM
    test_qmmm = ForceQMMM(at0, qm_mask, qm_calc, mm_calc, buffer_width=3.61)
    assert not np.count_nonzero(qmmm.qm_selection_mask) == np.count_nonzero(test_qmmm.qm_selection_mask)
    test_qmmm.set_masks_from_region(region)
    assert all(test_qmmm.qm_selection_mask == qmmm.qm_selection_mask)
    assert all(test_qmmm.qm_buffer_mask == qmmm.qm_buffer_mask)
    test_region = test_qmmm.get_region_from_masks(at0)
    assert all(region == test_region)