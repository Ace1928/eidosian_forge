from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
class NotPU21Representation:
    """
    Returned by is_pu_2_1_representation if cross ratios do not fulfill
    conditions to be a PU(2,1)-representation.
    Contains the reason why cross ratios fail to do so.
    Cast to bool evaluates to False.
    """

    def __init__(self, reason):
        self.reason = reason

    def __repr__(self):
        return 'NotPU21Representation(reason = %r)' % self.reason

    def __bool__(self):
        return False
    __nonzero__ = __bool__