import math
import numpy
from rdkit import Chem
from rdkit.Chem import Graphs, rdchem, rdMolDescriptors
from rdkit.ML.InfoTheory import entropy
def AvgIpc(mol, dMat=None, forceDMat=False):
    """This returns the average information content of the coefficients of the characteristic
    polynomial of the adjacency matrix of a hydrogen-suppressed graph of a molecule.

    From Eq 7 of D. Bonchev & N. Trinajstic, J. Chem. Phys. vol 67, 4517-4533 (1977)

  """
    return Ipc(mol, avg=True, dMat=dMat, forceDMat=forceDMat)