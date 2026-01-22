import base64
import hashlib
import logging
import os
import re
import tempfile
import uuid
from collections import namedtuple
from rdkit import Chem, RDConfig
from rdkit.Chem.MolKey import InchiInfo
def CheckCTAB(ctab, isSmiles=True):
    if not __initCalled:
        initStruchk()
    mol_str = ctab
    if not mol_str:
        raise BadMoleculeException('Unexpected blank or NULL molecule')
    mol_str = _fix_line_ends(mol_str)
    mol_str = _fix_chemdraw_header(mol_str)
    if isSmiles:
        if mol_str and NULL_SMILES_RE.match(mol_str):
            return T_NULL_MOL
        return pyAvalonTools.CheckMoleculeString(mol_str, isSmiles)
    ctab_lines = mol_str.split('\n')
    if len(ctab_lines) <= 3:
        raise BadMoleculeException('Not enough lines in CTAB')
    _ctab_remove_chiral_flag(ctab_lines)
    if not _ctab_has_atoms(ctab_lines):
        return T_NULL_MOL
    mol_str = '\n'.join(ctab_lines)
    return pyAvalonTools.CheckMoleculeString(mol_str, isSmiles)