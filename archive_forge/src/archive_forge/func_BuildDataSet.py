import csv
import pickle
import re
import numpy
from rdkit import RDRandom as random
from rdkit.DataStructs import BitUtils
from rdkit.ML.Data import MLData
from rdkit.utils import fileutils
def BuildDataSet(fileName):
    """ builds a data set from a .dat file

      **Arguments**

        - fileName: the name of the .dat file

      **Returns**

        an _MLData.MLDataSet_

    """
    with open(fileName, 'r') as inFile:
        varNames, qBounds = ReadVars(inFile)
        ptNames, examples = ReadGeneralExamples(inFile)
    data = MLData.MLDataSet(examples, qBounds=qBounds, varNames=varNames, ptNames=ptNames)
    return data