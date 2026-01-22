import os
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdChemReactions
def _uniqueOnly(lst):
    seen = []
    ps = Chem.SmilesWriteParams()
    cxflags = Chem.CXSmilesFields.CX_ENHANCEDSTEREO
    for entry in lst:
        if entry:
            smi = '.'.join(sorted([Chem.MolToCXSmiles(x, ps, cxflags) for x in entry]))
            if smi not in seen:
                seen.append(smi)
                yield entry