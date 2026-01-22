import numpy
from rdkit.ML.DecTree import DecTree
from rdkit.ML.InfoTheory import entropy
def GenVarTable(examples, nPossibleVals, vars):
    """Generates a list of variable tables for the examples passed in.

    The table for a given variable records the number of times each possible value
    of that variable appears for each possible result of the function.

  **Arguments**

    - examples: a list (nInstances long) of lists of variable values + instance
              values

    - nPossibleVals: a list containing the number of possible values of
                   each variable + the number of values of the function.

    - vars:  a list of the variables to include in the var table


  **Returns**

      a list of variable result tables. Each table is a Numeric array
        which is varValues x nResults
  """
    nVars = len(vars)
    res = [None] * nVars
    nFuncVals = nPossibleVals[-1]
    for i in range(nVars):
        res[i] = numpy.zeros((nPossibleVals[vars[i]], nFuncVals), 'i')
    for example in examples:
        val = int(example[-1])
        for i in range(nVars):
            res[i][int(example[vars[i]]), val] += 1
    return res