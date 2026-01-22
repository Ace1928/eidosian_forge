from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
class PtolemyCannotBeCheckedError(Exception):

    def __init__(self):
        msg = 'Use .cross_ratios().check_against_manifold(...) since checking Ptolemy coordinates for non-trivial generalized obstruction class is not supported.'
        Exception.__init__(self, msg)