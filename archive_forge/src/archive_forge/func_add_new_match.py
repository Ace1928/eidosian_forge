import sys
import copy
import itertools
import re
import time
import weakref
from collections import Counter, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain, combinations
def add_new_match(self, subgraph, mol, smarts):
    sizes = self.sizes
    num_subgraph_bonds = len(subgraph.bond_indices)
    if num_subgraph_bonds < sizes[1]:
        return sizes
    num_subgraph_atoms = len(subgraph.atom_indices)
    if num_subgraph_bonds == sizes[1] and num_subgraph_atoms <= sizes[0]:
        return sizes
    if check_completeRingsOnly(smarts, subgraph, mol):
        return self._new_best(num_subgraph_atoms, num_subgraph_bonds, smarts)
    return sizes