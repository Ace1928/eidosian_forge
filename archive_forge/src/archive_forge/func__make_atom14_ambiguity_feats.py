import collections
import copy
import functools
from importlib import resources
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
def _make_atom14_ambiguity_feats() -> None:
    for res, pairs in residue_atom_renaming_swaps.items():
        res_idx = restype_order[restype_3to1[res]]
        for atom1, atom2 in pairs.items():
            atom1_idx = restype_name_to_atom14_names[res].index(atom1)
            atom2_idx = restype_name_to_atom14_names[res].index(atom2)
            restype_atom14_ambiguous_atoms[res_idx, atom1_idx] = 1
            restype_atom14_ambiguous_atoms[res_idx, atom2_idx] = 1
            restype_atom14_ambiguous_atoms_swap_idx[res_idx, atom1_idx] = atom2_idx
            restype_atom14_ambiguous_atoms_swap_idx[res_idx, atom2_idx] = atom1_idx