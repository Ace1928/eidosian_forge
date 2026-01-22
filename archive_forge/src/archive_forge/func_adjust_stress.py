import warnings
import numpy as np
from ase.constraints import FixConstraint
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.utils import atoms_to_spglib_cell
def adjust_stress(self, atoms, stress):
    raw_stress = voigt_6_to_full_3x3_stress(stress)
    symmetrized_stress = symmetrize_rank2(atoms.get_cell(), atoms.cell.reciprocal().T, raw_stress, self.rotations)
    stress[:] = full_3x3_to_voigt_6_stress(symmetrized_stress)