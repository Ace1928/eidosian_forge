import math
import numpy
def PrincipalComponents(mat, reverseOrder=1):
    """ do a principal components analysis

  """
    covMat = FormCorrelationMatrix(mat)
    eigenVals, eigenVects = numpy.linalg.eig(covMat)
    eigenVals = getattr(eigenVals, 'real', eigenVals)
    eigenVects = getattr(eigenVects, 'real', eigenVects)
    ptOrder = numpy.argsort(eigenVals).tolist()
    if reverseOrder:
        ptOrder.reverse()
    eigenVals = numpy.array([eigenVals[x] for x in ptOrder])
    eigenVects = numpy.array([eigenVects[x] for x in ptOrder])
    return (eigenVals, eigenVects)