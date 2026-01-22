import pickle
import numpy
from rdkit.ML.Data import DataUtils
def SortModels(self, sortOnError=True):
    """ sorts the list of models

      **Arguments**

        sortOnError: toggles sorting on the models' errors rather than their counts


    """
    if sortOnError:
        order = numpy.argsort(self.errList)
    else:
        order = numpy.argsort(self.countList)
    self.modelList = [self.modelList[x] for x in order]
    self.countList = [self.countList[x] for x in order]
    self.errList = [self.errList[x] for x in order]