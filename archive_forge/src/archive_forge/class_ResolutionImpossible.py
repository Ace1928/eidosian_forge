import collections
import itertools
import operator
from .providers import AbstractResolver
from .structs import DirectedGraph, IteratorMapping, build_iter_view
class ResolutionImpossible(ResolutionError):

    def __init__(self, causes):
        super(ResolutionImpossible, self).__init__(causes)
        self.causes = causes