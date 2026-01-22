from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def all_normalized_translations(self):
    """
        Compute the translations corresponding to the meridian and longitude
        for each cusp.
        """
    self.compute_translations()
    return [ComplexCuspCrossSection._get_normalized_translations(vertex) for vertex in self.mcomplex.Vertices]