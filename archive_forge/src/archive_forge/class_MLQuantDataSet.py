import copy
import math
import numpy
class MLQuantDataSet(MLDataSet):
    """ a data set for holding quantized data


      **Note**

        this is intended to be a read-only data structure
        (i.e. after calling the constructor you cannot touch it)

      **Big differences to MLDataSet**

        1) data are stored in a numpy array since they are homogenous

        2) results are assumed to be quantized (i.e. no qBounds entry is required)

    """

    def _CalcNPossible(self, data):
        """calculates the number of possible values of each variable

          **Arguments**

             -data: a list of examples to be used

          **Returns**

             a list of nPossible values for each variable

        """
        return [max(x) + 1 for x in numpy.transpose(data)]

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

    def GetAllData(self):
        """ returns a *copy* of the data

        """
        return self.data.tolist()

    def GetInputData(self):
        """ returns the input data

         **Note**

           _inputData_ means the examples without their result fields
            (the last _NResults_ entries)

        """
        return self.data[:, :-self.nResults].tolist()

    def GetResults(self):
        """ Returns the result fields from each example

        """
        if self.GetNResults() > 1:
            v = self.GetNResults()
            res = [x[-v:] for x in self.data]
        else:
            res = [x[-1] for x in self.data]
        return res

    def __init__(self, data, nVars=None, nPts=None, nPossibleVals=None, qBounds=None, varNames=None, ptNames=None, nResults=1):
        """ Constructor

          **Arguments**

            - data: a list of lists containing the data. The data are copied, so don't worry
                  about us overwriting them.

            - nVars: the number of variables

            - nPts: the number of points

            - nPossibleVals: an list containing the number of possible values
                           for each variable (should contain 0 when not relevant)
                           This is _nVars_ long

            - qBounds: a list of lists containing quantization bounds for variables
                     which are to be quantized (note, this class does not quantize
                     the variables itself, it merely stores quantization bounds.
                     an empty sublist indicates no quantization for a given variable
                     This is _nVars_ long

            - varNames: a list of the names of the variables.
                     This is _nVars_ long

            - ptNames: the names (labels) of the individual data points
               This is _nPts_ long

            - nResults: the number of results columns in the data lists.  This is usually
                        1, but can be higher.
        """
        self.data = numpy.array(data)
        self.nResults = nResults
        if nVars is None:
            nVars = len(data[0]) - self.nResults
        self.nVars = nVars
        if nPts is None:
            nPts = len(data)
        self.nPts = nPts
        if qBounds is None:
            qBounds = [[]] * self.nVars
        self.qBounds = qBounds
        if nPossibleVals is None:
            nPossibleVals = self._CalcNPossible(data)
        self.nPossibleVals = nPossibleVals
        if varNames is None:
            varNames = [''] * self.nVars
        self.varNames = varNames
        if ptNames is None:
            ptNames = [''] * self.nPts
        self.ptNames = ptNames