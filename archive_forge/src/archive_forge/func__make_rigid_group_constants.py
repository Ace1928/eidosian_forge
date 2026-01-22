import collections
import copy
import functools
from importlib import resources
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
def _make_rigid_group_constants() -> None:
    """Fill the arrays above."""
    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        for atomname, group_idx, atom_position in rigid_group_atom_positions[resname]:
            atomtype = atom_order[atomname]
            restype_atom37_to_rigid_group[restype, atomtype] = group_idx
            restype_atom37_mask[restype, atomtype] = 1
            restype_atom37_rigid_group_positions[restype, atomtype, :] = atom_position
            atom14idx = restype_name_to_atom14_names[resname].index(atomname)
            restype_atom14_to_rigid_group[restype, atom14idx] = group_idx
            restype_atom14_mask[restype, atom14idx] = 1
            restype_atom14_rigid_group_positions[restype, atom14idx, :] = atom_position
    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        atom_positions: Dict[str, np.ndarray] = {name: np.array(pos) for name, _, pos in rigid_group_atom_positions[resname]}
        restype_rigid_group_default_frame[restype, 0, :, :] = np.eye(4)
        restype_rigid_group_default_frame[restype, 1, :, :] = np.eye(4)
        mat = _make_rigid_transformation_4x4(ex=atom_positions['N'] - atom_positions['CA'], ey=np.array([1.0, 0.0, 0.0]), translation=atom_positions['N'])
        restype_rigid_group_default_frame[restype, 2, :, :] = mat
        mat = _make_rigid_transformation_4x4(ex=atom_positions['C'] - atom_positions['CA'], ey=atom_positions['CA'] - atom_positions['N'], translation=atom_positions['C'])
        restype_rigid_group_default_frame[restype, 3, :, :] = mat
        if chi_angles_mask[restype][0]:
            base_atom_names = chi_angles_atoms[resname][0]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            mat = _make_rigid_transformation_4x4(ex=base_atom_positions[2] - base_atom_positions[1], ey=base_atom_positions[0] - base_atom_positions[1], translation=base_atom_positions[2])
            restype_rigid_group_default_frame[restype, 4, :, :] = mat
        for chi_idx in range(1, 4):
            if chi_angles_mask[restype][chi_idx]:
                axis_end_atom_name = chi_angles_atoms[resname][chi_idx][2]
                axis_end_atom_position = atom_positions[axis_end_atom_name]
                mat = _make_rigid_transformation_4x4(ex=axis_end_atom_position, ey=np.array([-1.0, 0.0, 0.0]), translation=axis_end_atom_position)
                restype_rigid_group_default_frame[restype, 4 + chi_idx, :, :] = mat