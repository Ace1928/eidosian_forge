import collections
import itertools
import operator
from .providers import AbstractResolver
from .structs import DirectedGraph, IteratorMapping, build_iter_view
class RequirementsConflicted(ResolverException):

    def __init__(self, criterion):
        super(RequirementsConflicted, self).__init__(criterion)
        self.criterion = criterion

    def __str__(self):
        return 'Requirements conflict: {}'.format(', '.join((repr(r) for r in self.criterion.iter_requirement())))