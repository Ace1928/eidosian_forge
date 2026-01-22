import os.path
import re
from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import (GetAtomPairFingerprint,
def GetBPFingerprint(mol, fpfn=GetAtomPairFingerprint):
    """
    >>> from rdkit import Chem
    >>> fp = GetBPFingerprint(Chem.MolFromSmiles('OCC(=O)O'))
    >>> fp.GetTotalVal()
    10
    >>> nze = fp.GetNonzeroElements()
    >>> sorted([(k, v) for k, v in nze.items()])
    [(32834, 1), (49219, 2), (98370, 2), (98401, 1), (114753, 2), (114786, 1), (114881, 1)]

    """
    typs = [typMap[x] for x in AssignPattyTypes(mol)]
    return fpfn(mol, atomInvariants=typs)