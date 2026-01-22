from .sage_helper import _within_sage
from .pari import *
import re
class MorphismToSPN(Morphism):

    def __init__(self, source, target, precision):
        Morphism.__init__(self, Hom(source, target, Rings()))
        self.SPN = target
        self.target_precision = precision

    def _call_(self, x):
        result = Number(x, precision=self.SPN.precision())
        result._precision = self.target_precision
        return result