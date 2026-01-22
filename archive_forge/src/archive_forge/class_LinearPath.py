import logging
import math
import numpy as np
from ase.utils import longsum
class LinearPath:
    """Describes a linear search path of the form t -> t g
    """

    def __init__(self, dirn):
        """Initialise LinearPath object

        Args:
           dirn : search direction
        """
        self.dirn = dirn

    def step(self, alpha):
        return alpha * self.dirn