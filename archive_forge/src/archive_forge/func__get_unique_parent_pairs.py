import numpy as np
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.Selection import unfold_entities, entity_levels, uniqueify
def _get_unique_parent_pairs(self, pair_list):
    parent_pair_list = []
    for e1, e2 in pair_list:
        p1 = e1.get_parent()
        p2 = e2.get_parent()
        if p1 == p2:
            continue
        elif p1 < p2:
            parent_pair_list.append((p1, p2))
        else:
            parent_pair_list.append((p2, p1))
    return uniqueify(parent_pair_list)