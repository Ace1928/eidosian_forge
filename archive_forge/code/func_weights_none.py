import math
from collections import namedtuple
from rdkit import Chem
from rdkit.Chem import Crippen, MolSurf
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem.ChemUtils.DescriptorUtilities import setDescriptorVersion
def weights_none(mol):
    """
  Calculates the QED descriptor using unit weights.
  """
    return qed(mol, w=WEIGHT_NONE)