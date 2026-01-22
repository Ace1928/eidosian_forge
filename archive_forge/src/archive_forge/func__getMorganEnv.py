import os
import warnings
from collections import namedtuple
from importlib.util import find_spec
from io import BytesIO
import numpy
from rdkit import Chem
from rdkit import RDConfig
from rdkit import rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from rdkit.Chem.Draw.MolDrawing import MolDrawing
from rdkit.Chem.Draw.rdMolDraw2D import *
def _getMorganEnv(mol, atomId, radius, baseRad, aromaticColor, ringColor, centerColor, extraColor, **kwargs):
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    bitPath = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomId)
    atomsToUse = set((atomId,))
    for b in bitPath:
        atomsToUse.add(mol.GetBondWithIdx(b).GetBeginAtomIdx())
        atomsToUse.add(mol.GetBondWithIdx(b).GetEndAtomIdx())
    enlargedEnv = set()
    for atom in atomsToUse:
        a = mol.GetAtomWithIdx(atom)
        for b in a.GetBonds():
            bidx = b.GetIdx()
            if bidx not in bitPath:
                enlargedEnv.add(bidx)
    enlargedEnv = list(enlargedEnv)
    enlargedEnv += bitPath
    amap = {}
    if enlargedEnv:
        submol = Chem.PathToSubmol(mol, enlargedEnv, atomMap=amap)
    else:
        submol = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, atomsToUse=atomsToUse))
    Chem.FastFindRings(submol)
    conf = Chem.Conformer(submol.GetNumAtoms())
    confOri = mol.GetConformer(0)
    for i1, i2 in amap.items():
        conf.SetAtomPosition(i2, confOri.GetAtomPosition(i1))
    submol.AddConformer(conf)
    envSubmol = []
    for i1, i2 in amap.items():
        for b in bitPath:
            beginAtom = amap[mol.GetBondWithIdx(b).GetBeginAtomIdx()]
            endAtom = amap[mol.GetBondWithIdx(b).GetEndAtomIdx()]
            envSubmol.append(submol.GetBondBetweenAtoms(beginAtom, endAtom).GetIdx())
    atomcolors, bondcolors = ({}, {})
    highlightAtoms, highlightBonds = ([], [])
    highlightRadii = {}
    for aidx in amap.keys():
        if aidx in atomsToUse:
            color = None
            if centerColor and aidx == atomId:
                color = centerColor
            elif aromaticColor and mol.GetAtomWithIdx(aidx).GetIsAromatic():
                color = aromaticColor
            elif ringColor and mol.GetAtomWithIdx(aidx).IsInRing():
                color = ringColor
            if color is not None:
                atomcolors[amap[aidx]] = color
                highlightAtoms.append(amap[aidx])
                highlightRadii[amap[aidx]] = baseRad
        else:
            submol.GetAtomWithIdx(amap[aidx]).SetAtomicNum(0)
            submol.GetAtomWithIdx(amap[aidx]).UpdatePropertyCache()
    color = extraColor
    for bid in submol.GetBonds():
        bidx = bid.GetIdx()
        if bidx not in envSubmol:
            bondcolors[bidx] = color
            highlightBonds.append(bidx)
    return FingerprintEnv(submol, highlightAtoms, atomcolors, highlightBonds, bondcolors, highlightRadii)