import csv
import pickle
import re
import numpy
from rdkit import RDRandom as random
from rdkit.DataStructs import BitUtils
from rdkit.ML.Data import MLData
from rdkit.utils import fileutils
def ReadQuantExamples(inFile):
    """ reads the examples from a .qdat file

      **Arguments**

        - inFile: a file object

      **Returns**

        a 2-tuple containing:

          1) the names of the examples

          2) a list of lists containing the examples themselves

      **Note**

        because this is reading a .qdat file, it assumed that all variable values
        are integers

    """
    expr1 = re.compile('^#')
    expr2 = re.compile('[ ]+|[\\t]+')
    examples = []
    names = []
    inLine = inFile.readline()
    while inLine:
        if expr1.search(inLine) is None:
            resArr = expr2.split(inLine)
            if len(resArr) > 1:
                examples.append([int(x) for x in resArr[1:]])
                names.append(resArr[0])
        inLine = inFile.readline()
    return (names, examples)