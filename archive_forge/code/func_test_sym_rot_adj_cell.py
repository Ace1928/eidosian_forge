import numpy as np
import pytest
from ase.atoms import Atoms
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.calculators.lj import LennardJones
from ase.spacegroup.symmetrize import FixSymmetry, check_symmetry, is_subgroup
from ase.optimize.precon.lbfgs import PreconLBFGS
from ase.constraints import UnitCellFilter, ExpCellFilter
@pytest.mark.filterwarnings('ignore:ASE Atoms-like input is deprecated')
@pytest.mark.filterwarnings('ignore:Armijo linesearch failed')
def test_sym_rot_adj_cell(filter):
    print('SYM POS+CELL ROT')
    at_init, at_rot = setup_cell()
    at_sym_3_rot = at_init.copy()
    at_sym_3_rot.set_constraint(FixSymmetry(at_sym_3_rot, adjust_positions=True, adjust_cell=True))
    di, df = symmetrized_optimisation(at_sym_3_rot, filter)
    assert di['number'] == 229 and is_subgroup(sub_data=di, sup_data=df)