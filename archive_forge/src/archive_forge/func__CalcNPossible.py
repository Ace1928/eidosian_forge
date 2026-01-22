import copy
import math
import numpy
def _CalcNPossible(self, data):
    """calculates the number of possible values of each variable

          **Arguments**

             -data: a list of examples to be used

          **Returns**

             a list of nPossible values for each variable

        """
    return [max(x) + 1 for x in numpy.transpose(data)]