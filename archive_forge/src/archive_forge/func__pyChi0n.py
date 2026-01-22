import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _pyChi0n(mol):
    """  Similar to Hall Kier Chi0v, but uses nVal instead of valence
  This makes a big difference after we get out of the first row.

  """
    deltas = [_nVal(x) for x in mol.GetAtoms()]
    while deltas.count(0):
        deltas.remove(0)
    deltas = numpy.array(deltas, 'd')
    res = sum(numpy.sqrt(1.0 / deltas))
    return res