from .component import NonZeroDimensionalComponent
from . import processFileBase
from . import processRurFile
from . import utilities
from . import coordinates
from .polynomial import Polynomial
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
class SolutionContainer:

    def __init__(self, solutions):
        self._solutions = solutions

    def solutions(self, numerical=False):
        if numerical:
            return self._solutions.numerical()
        else:
            return self._solutions

    def number_field(self):
        return self._solutions.number_field()