import numpy
from rdkit.ML.InfoTheory import entropy
def CalcInfoGains(bitVects, actVals, nPossibleActs, nPossibleBitVals=2):
    """  Calculates the information gain for a set of points and activity values

  **Arguments**

    - bitVects: a *sequence* containing *IntVectors*

    - actVals: a *sequence*

    - nPossibleActs: the (integer) number of possible activity values.

    - nPossibleBitVals: (optional) if specified, this integer provides the maximum
      value attainable by the (increasingly inaccurately named) bits in _bitVects_.

   **Returns**

     a list of floats

  """
    if len(bitVects) != len(actVals):
        raise ValueError('var and activity lists should be the same length')
    nBits = len(bitVects[0])
    res = numpy.zeros(nBits, float)
    for bit in range(nBits):
        counts = FormCounts(bitVects, actVals, bit, nPossibleActs, nPossibleBitVals=nPossibleBitVals)
        res[bit] = entropy.InfoGain(counts)
    return res