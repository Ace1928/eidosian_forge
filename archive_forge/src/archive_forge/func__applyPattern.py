import os
import re
from collections import namedtuple
from contextlib import closing
from rdkit import Chem, RDConfig
from rdkit.Chem.rdmolfiles import SDMolSupplier, SmilesMolSupplier
def _applyPattern(m, salt, notEverything):
    nAts = m.GetNumAtoms()
    if not nAts:
        return m
    res = m
    t = Chem.DeleteSubstructs(res, salt, True)
    if not t or (notEverything and t.GetNumAtoms() == 0):
        return res
    res = t
    while res.GetNumAtoms() and nAts > res.GetNumAtoms():
        nAts = res.GetNumAtoms()
        t = Chem.DeleteSubstructs(res, salt, True)
        if notEverything and t.GetNumAtoms() == 0:
            break
        res = t
    return res