import copy
import random
import numpy
from rdkit.DataStructs.VectCollection import VectCollection
from rdkit.ML import InfoTheory
from rdkit.ML.DecTree import SigTree
def _GenerateRandomEnsemble(nToInclude, nBits):
    """  Generates a random subset of a group of indices

  **Arguments**

    - nToInclude: the size of the desired set

    - nBits: the maximum index to be included in the set

   **Returns**

     a list of indices

  """
    return random.sample(range(nBits), nToInclude)