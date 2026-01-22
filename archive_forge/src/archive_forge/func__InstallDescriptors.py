import bisect
import numpy
from rdkit.Chem.EState.EState import EStateIndices as EStateIndices_
from rdkit.Chem.MolSurf import _LabuteHelper as VSAContribs_
def _InstallDescriptors():
    for nbin in range(len(vsaBins) + 1):
        name, fn = _descriptor_VSA_EState(nbin)
        globals()[name] = fn
    for nbin in range(len(estateBins) + 1):
        name, fn = _descriptor_EState_VSA(nbin)
        globals()[name] = fn