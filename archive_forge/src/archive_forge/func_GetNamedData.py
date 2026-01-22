import copy
import math
import numpy
def GetNamedData(self):
    """ returns a list of named examples

         **Note**

           a named example is the result of prepending the example
            name to the data list

        """
    res = [None] * self.nPts
    for i in range(self.nPts):
        res[i] = [self.ptNames[i]] + self.data[i].tolist()
    return res