import enum
import hashlib
import json
import logging
import re
from typing import Iterable, Optional
from rdkit import Chem
from rdkit.Chem import rdMolHash
def _CanonicalizeSRUSGroup(mol, sg, atRanks, bndOrder, sortAtomAndBondOrder):
    """
    NOTES: if sortAtomAndBondOrder is true then the atom and bond lists will be sorted.
    This assumes that the ordering of those lists is not important

    """
    if sg.GetProp('TYPE') != 'SRU':
        return None
    ats = tuple((atRanks[x] for x in sg.GetAtoms()))
    bnds = tuple((bndOrder[x] for x in sg.GetBonds()))
    if sortAtomAndBondOrder:
        ats = tuple(sorted(ats))
        bnds = tuple(sorted(bnds))
    props = sg.GetPropsAsDict()
    res = dict(type='SRU', atoms=ats, bonds=bnds, index=props.get('index', 0), connect=props.get('CONNECT', ''), label=props.get('LABEL', ''))
    if 'PARENT' in props:
        res['PARENT'] = props['PARENT']
    if 'XBHEAD' in props:
        xbhbonds = tuple((_GetCanononicalBondRep(mol.GetBondWithIdx(x), atRanks) for x in props['XBHEAD']))
        if sortAtomAndBondOrder:
            xbhbonds = tuple(sorted(xbhbonds))
        res['XBHEAD'] = xbhbonds
    if 'XBCORR' in props:
        xbcorrbonds = tuple((_GetCanononicalBondRep(mol.GetBondWithIdx(x), atRanks) for x in props['XBCORR']))
        if len(xbcorrbonds) % 2:
            raise ValueError('XBCORR should have 2N bonds')
        if sortAtomAndBondOrder:
            tmp = []
            for i in range(0, len(xbcorrbonds), 2):
                b1, b2 = (xbcorrbonds[i], xbcorrbonds[i + 1])
                if b1 > b2:
                    b1, b2 = (b2, b1)
                tmp.append((b1, b2))
            xbcorrbonds = tuple(sorted(tmp))
        res['XBCORR'] = xbcorrbonds
    return res