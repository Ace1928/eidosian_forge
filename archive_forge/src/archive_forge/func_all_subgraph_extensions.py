import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def all_subgraph_extensions(enumeration_mol, subgraph, visited_bond_indices, internal_bonds, external_edges):
    if not external_edges:
        it = nonempty_powerset(internal_bonds)
        for internal_bond in it:
            bond_indices = set(subgraph.bond_indices)
            bond_indices.update(internal_bond)
            yield (None, Subgraph(subgraph.atom_indices, frozenset(bond_indices)), 0, 0)
        return
    if not internal_bonds:
        it = nonempty_powerset(external_edges)
        exclude_bonds = set(chain(visited_bond_indices, (edge.bond_index for edge in external_edges)))
        for external_ext in it:
            new_atoms = frozenset((ext.end_atom_index for ext in external_ext))
            atom_indices = frozenset(chain(subgraph.atom_indices, new_atoms))
            bond_indices = frozenset(chain(subgraph.bond_indices, (ext.bond_index for ext in external_ext)))
            num_possible_atoms, num_possible_bonds = find_extension_size(enumeration_mol, new_atoms, exclude_bonds, external_ext)
            yield (new_atoms, Subgraph(atom_indices, bond_indices), num_possible_atoms, num_possible_bonds)
        return
    internal_powerset = list(powerset(internal_bonds))
    external_powerset = powerset(external_edges)
    exclude_bonds = set(chain(visited_bond_indices, (edge.bond_index for edge in external_edges)))
    for external_ext in external_powerset:
        if not external_ext:
            for internal_bond in internal_powerset[1:]:
                bond_indices = set(subgraph.bond_indices)
                bond_indices.update(internal_bond)
                yield (None, Subgraph(subgraph.atom_indices, bond_indices), 0, 0)
        else:
            new_atoms = frozenset((ext.end_atom_index for ext in external_ext))
            atom_indices = frozenset(chain(subgraph.atom_indices, new_atoms))
            bond_indices = frozenset(chain(subgraph.bond_indices, (ext.bond_index for ext in external_ext)))
            num_possible_atoms, num_possible_bonds = find_extension_size(enumeration_mol, atom_indices, exclude_bonds, external_ext)
            for internal_bond in internal_powerset:
                bond_indices2 = frozenset(chain(bond_indices, internal_bond))
                yield (new_atoms, Subgraph(atom_indices, bond_indices2), num_possible_atoms, num_possible_bonds)