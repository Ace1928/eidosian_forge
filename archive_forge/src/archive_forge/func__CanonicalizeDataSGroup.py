import enum
import hashlib
import json
import logging
import re
from typing import Iterable, Optional
from rdkit import Chem
from rdkit.Chem import rdMolHash
def _CanonicalizeDataSGroup(sg, atRanks, bndOrder, fieldNames=('Atrop',), sortAtomAndBondOrder=True):
    """
    NOTES: if sortAtomAndBondOrder is true then the atom and bond lists will
    be sorted. This assumes that the order of the atoms in that list is not
    important

    """
    if sg.GetProp('TYPE') != 'DAT' or not sg.HasProp('FIELDNAME'):
        return None
    fieldName = sg.GetProp('FIELDNAME')
    if fieldName not in fieldNames:
        return None
    data = sg.GetStringVectProp('DATAFIELDS')
    if len(data) > 1:
        raise ValueError('cannot canonicalize data groups with multiple data fields')
    data = data[0]
    ats = tuple((atRanks[x] for x in sg.GetAtoms()))
    if sortAtomAndBondOrder:
        ats = tuple(sorted(ats))
    bnds = tuple((bndOrder[x] for x in sg.GetBonds()))
    if sortAtomAndBondOrder:
        bnds = tuple(sorted(bnds))
    res = dict(fieldName=fieldName, atom=ats, bonds=bnds, value=data)
    return res