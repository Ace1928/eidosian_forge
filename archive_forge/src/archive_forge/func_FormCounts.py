import numpy
from rdkit.ML.InfoTheory import entropy
def FormCounts(bitVects, actVals, whichBit, nPossibleActs, nPossibleBitVals=2):
    """ generates the counts matrix for a particular bit

  **Arguments**

    - bitVects: a *sequence* containing *IntVectors*

    - actVals: a *sequence*

    - whichBit: an integer, the bit number to use.

    - nPossibleActs: the (integer) number of possible activity values.

    - nPossibleBitVals: (optional) if specified, this integer provides the maximum
      value attainable by the (increasingly inaccurately named) bits in _bitVects_.

  **Returns**

    a Numeric array with the counts

  **Notes**

    This is really intended for internal use.

  """
    if len(bitVects) != len(actVals):
        raise ValueError('var and activity lists should be the same length')
    res = numpy.zeros((nPossibleBitVals, nPossibleActs), numpy.integer)
    for i in range(len(bitVects)):
        res[bitVects[i][whichBit], actVals[i]] += 1
    return res