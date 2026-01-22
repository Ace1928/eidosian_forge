import csv
import pickle
import re
import numpy
from rdkit import RDRandom as random
from rdkit.DataStructs import BitUtils
from rdkit.ML.Data import MLData
from rdkit.utils import fileutils
def ReadGeneralExamples(inFile):
    """ reads the examples from a .dat file

      **Arguments**

        - inFile: a file object

      **Returns**

        a 2-tuple containing:

          1) the names of the examples

          2) a list of lists containing the examples themselves

      **Note**

        - this attempts to convert variable values to ints, then floats.
          if those both fail, they are left as strings

    """
    expr1 = re.compile('^#')
    expr2 = re.compile('[ ]+|[\\t]+')
    examples = []
    names = []
    inLine = inFile.readline()
    while inLine:
        if expr1.search(inLine) is None:
            resArr = expr2.split(inLine)[:-1]
            if len(resArr) > 1:
                for i in range(1, len(resArr)):
                    d = resArr[i]
                    try:
                        resArr[i] = int(d)
                    except ValueError:
                        try:
                            resArr[i] = float(d)
                        except ValueError:
                            pass
                examples.append(resArr[1:])
                names.append(resArr[0])
        inLine = inFile.readline()
    return (names, examples)