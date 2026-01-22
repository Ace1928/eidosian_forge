import math
import numpy
def MeanAndDev(vect, sampleSD=1):
    """ returns the mean and standard deviation of a vector """
    vect = numpy.array(vect, 'd')
    n = vect.shape[0]
    if n <= 0:
        return (0.0, 0.0)
    mean = sum(vect) / n
    v = vect - mean
    if n > 1:
        if sampleSD:
            dev = numpy.sqrt(sum(v * v) / (n - 1))
        else:
            dev = numpy.sqrt(sum(v * v) / n)
    else:
        dev = 0
    return (mean, dev)