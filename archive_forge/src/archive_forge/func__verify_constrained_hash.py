import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
def _verify_constrained_hash(self):
    """
        Warn if the object name is not explicitly set.
        """
    changed_params = self.param.values(onlychanged=True)
    if self.time_dependent and 'name' not in changed_params:
        self.param.log(param.WARNING, 'Default object name used to set the seed: random values conditional on object instantiation order.')