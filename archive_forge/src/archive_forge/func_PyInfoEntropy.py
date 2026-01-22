import math
import numpy
def PyInfoEntropy(results):
    """ Calculates the informational entropy of a set of results.

  **Arguments**

    results is a 1D Numeric array containing the number of times a
    given set hits each possible result.
    For example, if a function has 3 possible results, and the
      variable in question hits them 5, 6 and 1 times each,
      results would be [5,6,1]

  **Returns**

    the informational entropy

  """
    nInstances = float(sum(results))
    if nInstances == 0:
        return 0
    probs = results / nInstances
    t = numpy.choose(numpy.greater(probs, 0.0), (1, probs))
    return sum(-probs * numpy.log(t) / _log2)