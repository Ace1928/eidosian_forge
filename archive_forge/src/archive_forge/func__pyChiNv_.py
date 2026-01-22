import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _pyChiNv_(mol, order=2):
    """  From equations (5),(15) and (16) of Rev. Comp. Chem. vol 2, 367-422, (1991)

  **NOTE**: because the current path finding code does, by design,
  detect rings as paths (e.g. in C1CC1 there is *1* atom path of
  length 3), values of ChiNv with N >= 3 may give results that differ
  from those provided by the old code in molecules that have rings of
  size 3.

  """
    deltas = numpy.array([1.0 / numpy.sqrt(hkd) if hkd != 0.0 else 0.0 for hkd in _hkDeltas(mol, skipHs=0)])
    accum = 0.0
    for path in Chem.FindAllPathsOfLengthN(mol, order + 1, useBonds=0):
        accum += numpy.prod(deltas[numpy.array(path)])
    return accum