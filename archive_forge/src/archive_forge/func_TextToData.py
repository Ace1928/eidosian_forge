import csv
import pickle
import re
import numpy
from rdkit import RDRandom as random
from rdkit.DataStructs import BitUtils
from rdkit.ML.Data import MLData
from rdkit.utils import fileutils
def TextToData(reader, ignoreCols=[], onlyCols=None):
    """ constructs  an _MLData.MLDataSet_ from a bunch of text
  #DOC
      **Arguments**
        - reader needs to be iterable and return lists of elements
          (like a csv.reader)

      **Returns**

         an _MLData.MLDataSet_

    """
    varNames = next(reader)
    if not onlyCols:
        keepCols = []
        for i, name in enumerate(varNames):
            if name not in ignoreCols:
                keepCols.append(i)
    else:
        keepCols = [-1] * len(onlyCols)
        for i, name in enumerate(varNames):
            if name in onlyCols:
                keepCols[onlyCols.index(name)] = i
    nCols = len(varNames)
    varNames = tuple([varNames[x] for x in keepCols])
    nVars = len(varNames)
    vals = []
    ptNames = []
    for splitLine in reader:
        if len(splitLine):
            if len(splitLine) != nCols:
                raise ValueError('unequal line lengths')
            tmp = [splitLine[x] for x in keepCols]
            ptNames.append(tmp[0])
            pt = [None] * (nVars - 1)
            for j in range(nVars - 1):
                try:
                    val = int(tmp[j + 1])
                except ValueError:
                    try:
                        val = float(tmp[j + 1])
                    except ValueError:
                        val = str(tmp[j + 1])
                pt[j] = val
            vals.append(pt)
    data = MLData.MLDataSet(vals, varNames=varNames, ptNames=ptNames)
    return data