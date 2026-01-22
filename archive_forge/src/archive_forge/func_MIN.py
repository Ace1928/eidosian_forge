from rdkit import RDConfig
from rdkit.ML.Descriptors import Descriptors, Parser
from rdkit.utils import chemutils
def MIN(self, desc, compos):
    """ *Calculator Method*

      minimum of the descriptor values across the composition

      **Arguments**

        - desc: the name of the descriptor

        - compos: the composition vector

      **Returns**

        a float

    """
    return min(map(lambda x, y=desc, z=self: z.atomDict[x[0]][y], compos))