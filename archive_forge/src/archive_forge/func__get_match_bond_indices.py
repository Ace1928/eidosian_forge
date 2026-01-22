import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def _get_match_bond_indices(pat, mol, match_atom_indices):
    bond_indices = []
    for bond in pat.GetBonds():
        mol_atom1 = match_atom_indices[bond.GetBeginAtomIdx()]
        mol_atom2 = match_atom_indices[bond.GetEndAtomIdx()]
        bond = mol.GetBondBetweenAtoms(mol_atom1, mol_atom2)
        assert bond is not None
        bond_indices.append(bond.GetIdx())
    return bond_indices