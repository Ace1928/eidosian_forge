import os.path
import re
from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import (GetAtomPairFingerprint,
def GetBTFingerprint(mol, fpfn=GetTopologicalTorsionFingerprint):
    """
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('OCC(N)O')
    >>> AssignPattyTypes(mol)
    ['POL', 'HYD', 'HYD', 'CAT', 'POL']
    >>> fp = GetBTFingerprint(mol)
    >>> fp.GetTotalVal()
    2
    >>> nze = fp.GetNonzeroElements()
    >>> sorted([(k, v) for k, v in nze.items()])
    [(538446850..., 1), (538446852..., 1)]

    """
    return GetBPFingerprint(mol, fpfn=fpfn)