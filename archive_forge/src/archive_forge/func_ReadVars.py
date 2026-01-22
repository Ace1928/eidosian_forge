import csv
import pickle
import re
import numpy
from rdkit import RDRandom as random
from rdkit.DataStructs import BitUtils
from rdkit.ML.Data import MLData
from rdkit.utils import fileutils
def ReadVars(inFile):
    """ reads the variables and quantization bounds from a .qdat or .dat file

      **Arguments**

        - inFile: a file object

      **Returns**

        a 2-tuple containing:

          1) varNames: a list of the variable names

          2) qbounds: the list of quantization bounds for each variable

    """
    varNames = []
    qBounds = []
    fileutils.MoveToMatchingLine(inFile, 'Variable Table')
    inLine = inFile.readline()
    while inLine.find('# ----') == -1:
        splitLine = inLine[2:].split('[')
        varNames.append(splitLine[0].strip())
        qBounds.append(splitLine[1][:-2])
        inLine = inFile.readline()
    for i in range(len(qBounds)):
        if qBounds[i] != '':
            l = qBounds[i].split(',')
            qBounds[i] = []
            for item in l:
                qBounds[i].append(float(item))
        else:
            qBounds[i] = []
    return (varNames, qBounds)