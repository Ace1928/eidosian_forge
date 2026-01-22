import math
import sys
import time
import numpy
import rdkit.DistanceGeometry as DG
from rdkit import Chem
from rdkit import RDLogger as logging
from rdkit.Chem import ChemicalFeatures, ChemicalForceFields
from rdkit.Chem import rdDistGeom as MolDG
from rdkit.Chem.Pharm3D import ExcludedVolume
from rdkit.ML.Data import Stats
def GetAtomHeavyNeighbors(atom):
    """ returns a list of the heavy-atom neighbors of the
  atom passed in:

  >>> m = Chem.MolFromSmiles('CCO')
  >>> l = GetAtomHeavyNeighbors(m.GetAtomWithIdx(0))
  >>> len(l)
  1
  >>> isinstance(l[0],Chem.Atom)
  True
  >>> l[0].GetIdx()
  1

  >>> l = GetAtomHeavyNeighbors(m.GetAtomWithIdx(1))
  >>> len(l)
  2
  >>> l[0].GetIdx()
  0
  >>> l[1].GetIdx()
  2

  """
    return [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() != 1]