import os
import re
import weakref
from rdkit import Chem, RDConfig
def _SetNodeBits(mol, node, res, idx):
    ms = mol.GetSubstructMatches(node.pattern)
    count = 0
    seen = {}
    for m in ms:
        if m[0] not in seen:
            count += 1
            seen[m[0]] = 1
    if count:
        res[idx] = count
        idx += 1
        for child in node.children:
            idx = _SetNodeBits(mol, child, res, idx)
    else:
        idx += len(node)
    return idx