import csv
import pickle
import re
import numpy
from rdkit import RDRandom as random
from rdkit.DataStructs import BitUtils
from rdkit.ML.Data import MLData
from rdkit.utils import fileutils
def BuildQuantDataSet(fileName):
    """ builds a data set from a .qdat file

      **Arguments**

        - fileName: the name of the .qdat file

      **Returns**

        an _MLData.MLQuantDataSet_

    """
    with open(fileName, 'r') as inFile:
        varNames, qBounds = ReadVars(inFile)
        ptNames, examples = ReadQuantExamples(inFile)
    data = MLData.MLQuantDataSet(examples, qBounds=qBounds, varNames=varNames, ptNames=ptNames)
    return data