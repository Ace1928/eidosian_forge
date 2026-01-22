import numpy
def EuclideanDistance(inData):
    """returns the euclidean metricMat between the points in _inData_

    **Arguments**

     - inData: a Numeric array of data points

    **Returns**

       a Numeric array with the metric matrix.  See the module documentation
       for the format.


  """
    nObjs = len(inData)
    res = numpy.zeros(nObjs * (nObjs - 1) / 2, float)
    nSoFar = 0
    for col in range(1, nObjs):
        for row in range(col):
            t = inData[row] - inData[col]
            res[nSoFar] = sum(t * t)
            nSoFar += 1
    return numpy.sqrt(res)