import pickle
import numpy
from rdkit.ML.Data import DataUtils
def SetDescriptorNames(self, names):
    """ registers the names of the descriptors this composite uses

      **Arguments**

       - names: a list of descriptor names (strings).

      **NOTE**

         the _names_ list is not
         copied, so if you modify it later, the composite itself will also be modified.

    """
    self._descNames = names