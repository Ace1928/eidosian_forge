import os
import numpy
from warnings import warn
from scipy.odr import __odrpack
def _sd2wt(self, sd):
    """ Convert standard deviation to weights.
        """
    return 1.0 / numpy.power(sd, 2)