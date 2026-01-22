from collections import \
import rdkit.Chem.ChemUtils.DescriptorUtilities as _du
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.EState.EState import (MaxAbsEStateIndex, MaxEStateIndex,
from rdkit.Chem.QED import qed
from rdkit.Chem.SpacialScore import SPS
def _FingerprintDensity(mol, func, *args, **kwargs):
    fp = func(*(mol,) + args, **kwargs)
    if hasattr(fp, 'GetNumOnBits'):
        val = fp.GetNumOnBits()
    else:
        val = len(fp.GetNonzeroElements())
    num_heavy_atoms = mol.GetNumHeavyAtoms()
    if num_heavy_atoms == 0:
        return 0
    return float(val) / num_heavy_atoms