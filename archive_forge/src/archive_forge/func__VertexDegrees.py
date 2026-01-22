import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def _VertexDegrees(mat, onlyOnes=0):
    """  *Internal Use Only*

  this is just a row sum of the matrix... simple, neh?

  """
    if not onlyOnes:
        res = sum(mat)
    else:
        res = sum(numpy.equal(mat, 1))
    return res