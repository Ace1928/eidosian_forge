import bisect
import numpy
from rdkit.Chem.EState.EState import EStateIndices as EStateIndices_
from rdkit.Chem.MolSurf import _LabuteHelper as VSAContribs_
def EState_VSA_(mol, bins=None, force=1):
    """ *Internal Use Only*
  """
    if not force and hasattr(mol, '_eStateVSA'):
        return mol._eStateVSA
    if bins is None:
        bins = estateBins
    propContribs = EStateIndices_(mol, force=force)
    volContribs = VSAContribs_(mol)
    ans = numpy.zeros(len(bins) + 1, dtype=numpy.float64)
    for i, prop in enumerate(propContribs):
        if prop is not None:
            nbin = bisect.bisect_right(bins, prop)
            ans[nbin] += volContribs[i + 1]
    mol._eStateVSA = ans
    return ans