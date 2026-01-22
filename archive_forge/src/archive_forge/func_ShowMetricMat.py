import numpy
def ShowMetricMat(metricMat, nObjs):
    """ displays a metric matrix

   **Arguments**

    - metricMat: the matrix to be displayed

    - nObjs: the number of objects to display

  """
    assert len(metricMat) == nObjs * (nObjs - 1) / 2, 'bad matrix length in FindMinValInList'
    for row in range(nObjs):
        for col in range(nObjs):
            if col <= row:
                print('   ---    ', end='')
            else:
                print('%10.6f' % metricMat[col * (col - 1) / 2 + row], end='')
        print()