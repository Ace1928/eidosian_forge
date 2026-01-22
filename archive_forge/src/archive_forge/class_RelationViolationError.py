from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
class RelationViolationError(Exception):
    """
    An exception raised when some supposed relation doesn't hold exactly
    or numerical.
    """

    def __init__(self, value, epsilon, comment):
        self.value = value
        self.epsilon = epsilon
        self.comment = comment

    def __str__(self):
        r = self.comment + ' is violated, '
        r += 'difference is %s' % self.value
        if self.epsilon is None:
            return r + ' (exact values)'
        return r + ' (epsilon = %s)' % self.epsilon