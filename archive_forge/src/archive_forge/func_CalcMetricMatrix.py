import numpy
def CalcMetricMatrix(inData, metricFunc):
    """ generates a metric matrix

    **Arguments**
     - inData is assumed to be a list of clusters (or anything with
       a GetPosition() method)

     - metricFunc is the function to be used to generate the matrix


    **Returns**

      the metric matrix as a Numeric array

  """
    inData = map(lambda x: x.GetPosition(), inData)
    return metricFunc(inData)