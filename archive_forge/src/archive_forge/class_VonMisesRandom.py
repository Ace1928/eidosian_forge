import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
class VonMisesRandom(RandomDistribution):
    """
    Circularly normal distributed random number.

    If kappa is zero, this distribution reduces to a uniform random
    angle over the range 0 to 2*pi.  Otherwise, it is concentrated to
    a greater or lesser degree (determined by kappa) around the mean
    mu.  For large kappa (narrow peaks), this distribution approaches
    the Gaussian (normal) distribution with variance 1/kappa.  See the
    random module for further details.
    """
    mu = param.Number(default=0.0, softbounds=(0.0, 2 * pi), doc='\n        Mean value, typically in the range 0 to 2*pi.')
    kappa = param.Number(default=1.0, bounds=(0.0, None), softbounds=(0.0, 50.0), doc='\n        Concentration (inverse variance).')

    def __call__(self):
        super().__call__()
        return self.random_generator.vonmisesvariate(self.mu, self.kappa)