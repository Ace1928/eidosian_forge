import numpy
from rdkit.ML.InfoTheory import entropy
def SparseRankBits(bitVects, actVals, metricFunc=AnalyzeSparseVects):
    """ Rank a set of bits according to a metric function

  **Arguments**

    - bitVects: a *sequence* containing SBVs

    - actVals: a *sequence*

    - metricFunc: (optional) the metric function to be used.  See _SparseCalcInfoGains()_
      for a description of the signature of this function.

   **Returns**

     A 2-tuple containing:

       - the relative order of the bits (a list of ints)

       - the metric calculated for each bit (a list of floats)

    **Notes**

      - these need to be bit vects and binary activities

  """
    info, metrics = metricFunc(bitVects, actVals)
    bitOrder = list(numpy.argsort(metrics))
    bitOrder.reverse()
    return (bitOrder, info)