import os
import re
from collections import namedtuple
from contextlib import closing
from rdkit import Chem, RDConfig
from rdkit.Chem.rdmolfiles import SDMolSupplier, SmilesMolSupplier
def _StripMol(self, mol, dontRemoveEverything=False, sanitize=True):

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
    StrippedMol = namedtuple('StrippedMol', ['mol', 'deleted'])
    deleted = []
    if dontRemoveEverything and len(Chem.GetMolFrags(mol)) <= 1:
        return StrippedMol(mol, deleted)
    modified = False
    natoms = mol.GetNumAtoms()
    for salt in self.salts:
        mol = _applyPattern(mol, salt, dontRemoveEverything)
        if natoms != mol.GetNumAtoms():
            natoms = mol.GetNumAtoms()
            modified = True
            deleted.append(salt)
            if dontRemoveEverything and len(Chem.GetMolFrags(mol)) <= 1:
                break
    if sanitize and modified and (mol.GetNumAtoms() > 0):
        Chem.SanitizeMol(mol)
    return StrippedMol(mol, deleted)