from rdkit import RDConfig
from rdkit.ML.Descriptors import Descriptors, Parser
from rdkit.utils import chemutils
def MAX(self, desc, compos):
    """ *Calculator Method*

      maximum of the descriptor values across the composition

      **Arguments**

        - desc: the name of the descriptor

        - compos: the composition vector

      **Returns**

        a float

    """
    return max(map(lambda x, y=desc, z=self: z.atomDict[x[0]][y], compos))