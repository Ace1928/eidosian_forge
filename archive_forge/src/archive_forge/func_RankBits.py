import numpy
from rdkit.ML.InfoTheory import entropy
def RankBits(bitVects, actVals, nPossibleBitVals=2, metricFunc=CalcInfoGains):
    """ Rank a set of bits according to a metric function

  **Arguments**

    - bitVects: a *sequence* containing *IntVectors*

    - actVals: a *sequence*

    - nPossibleBitVals: (optional) if specified, this integer provides the maximum
      value attainable by the (increasingly inaccurately named) bits in _bitVects_.

    - metricFunc: (optional) the metric function to be used.  See _CalcInfoGains()_
      for a description of the signature of this function.

   **Returns**

     A 2-tuple containing:

       - the relative order of the bits (a list of ints)

       - the metric calculated for each bit (a list of floats)

  """
    nPossibleActs = max(actVals) + 1
    metrics = metricFunc(bitVects, actVals, nPossibleActs, nPossibleBitVals=nPossibleBitVals)
    bitOrder = list(numpy.argsort(metrics))
    bitOrder.reverse()
    return (bitOrder, metrics)