import os
import abc
import numbers
import numpy as np
from . import polyutils as pu
@staticmethod
@abc.abstractmethod
def _der(c, m, scl):
    pass