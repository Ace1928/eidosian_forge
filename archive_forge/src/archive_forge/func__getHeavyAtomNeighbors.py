import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def _getHeavyAtomNeighbors(atom1, aid2=-1):
    """ Helper function to calculate the number of heavy atom neighbors.

      Arguments:
      - atom1:    the atom of interest
      - aid2:     atom index that should be excluded from neighbors (default: none)

      Return: a list of heavy atom neighbors of the given atom
  """
    if aid2 < 0:
        return [n for n in atom1.GetNeighbors() if n.GetSymbol() != 'H']
    else:
        return [n for n in atom1.GetNeighbors() if n.GetSymbol() != 'H' and n.GetIdx() != aid2]