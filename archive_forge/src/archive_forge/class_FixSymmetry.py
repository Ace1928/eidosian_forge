import warnings
import numpy as np
from ase.constraints import FixConstraint
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.utils import atoms_to_spglib_cell
class FixSymmetry(FixConstraint):
    """
    Constraint to preserve spacegroup symmetry during optimisation.

    Requires spglib package to be available.
    """

    def __init__(self, atoms, symprec=0.01, adjust_positions=True, adjust_cell=True, verbose=False):
        self.verbose = verbose
        refine_symmetry(atoms, symprec, self.verbose)
        sym = prep_symmetry(atoms, symprec, self.verbose)
        self.rotations, self.translations, self.symm_map = sym
        self.do_adjust_positions = adjust_positions
        self.do_adjust_cell = adjust_cell

    def adjust_cell(self, atoms, cell):
        if not self.do_adjust_cell:
            return
        cur_cell = atoms.get_cell()
        cur_cell_inv = atoms.cell.reciprocal().T
        delta_deform_grad = np.dot(cur_cell_inv, cell).T - np.eye(3)
        max_delta_deform_grad = np.max(np.abs(delta_deform_grad))
        if max_delta_deform_grad > 0.25:
            raise RuntimeError('FixSymmetry adjust_cell does not work properly with large deformation gradient step {} > 0.25'.format(max_delta_deform_grad))
        elif max_delta_deform_grad > 0.15:
            warnings.warn('FixSymmetry adjust_cell may be ill behaved with large deformation gradient step {}'.format(max_delta_deform_grad))
        symmetrized_delta_deform_grad = symmetrize_rank2(cur_cell, cur_cell_inv, delta_deform_grad, self.rotations)
        cell[:] = np.dot(cur_cell, (symmetrized_delta_deform_grad + np.eye(3)).T)

    def adjust_positions(self, atoms, new):
        if not self.do_adjust_positions:
            return
        step = new - atoms.positions
        symmetrized_step = symmetrize_rank1(atoms.get_cell(), atoms.cell.reciprocal().T, step, self.rotations, self.translations, self.symm_map)
        new[:] = atoms.positions + symmetrized_step

    def adjust_forces(self, atoms, forces):
        forces[:] = symmetrize_rank1(atoms.get_cell(), atoms.cell.reciprocal().T, forces, self.rotations, self.translations, self.symm_map)

    def adjust_stress(self, atoms, stress):
        raw_stress = voigt_6_to_full_3x3_stress(stress)
        symmetrized_stress = symmetrize_rank2(atoms.get_cell(), atoms.cell.reciprocal().T, raw_stress, self.rotations)
        stress[:] = full_3x3_to_voigt_6_stress(symmetrized_stress)

    def index_shuffle(self, atoms, ind):
        if len(atoms) != len(ind) or len(set(ind)) != len(ind):
            raise RuntimeError('FixSymmetry can only accomodate atom permutions, and len(Atoms) == {} != len(ind) == {} or ind has duplicates'.format(len(atoms), len(ind)))
        ind_reversed = np.zeros(len(ind), dtype=int)
        ind_reversed[ind] = range(len(ind))
        new_symm_map = []
        for sm in self.symm_map:
            new_sm = np.array([-1] * len(atoms))
            for at_i in range(len(ind)):
                new_sm[ind_reversed[at_i]] = ind_reversed[sm[at_i]]
            new_symm_map.append(new_sm)
        self.symm_map = new_symm_map