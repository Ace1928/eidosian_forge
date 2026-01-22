import bisect
from collections import defaultdict, namedtuple
from itertools import chain
import math
from operator import attrgetter, itemgetter
import random
import numpy
class selNSGA3WithMemory(object):
    """Class version of NSGA-III selection including memory for best, worst and
    extreme points. Registering this operator in a toolbox is a bit different
    than classical operators, it requires to instantiate the class instead
    of just registering the function::

        >>> from deap import base
        >>> ref_points = uniform_reference_points(nobj=3, p=12)
        >>> toolbox = base.Toolbox()
        >>> toolbox.register("select", selNSGA3WithMemory(ref_points))

    """

    def __init__(self, ref_points, nd='log'):
        self.ref_points = ref_points
        self.nd = nd
        self.best_point = numpy.full((1, ref_points.shape[1]), numpy.inf)
        self.worst_point = numpy.full((1, ref_points.shape[1]), -numpy.inf)
        self.extreme_points = None

    def __call__(self, individuals, k):
        chosen, memory = selNSGA3(individuals, k, self.ref_points, self.nd, self.best_point, self.worst_point, self.extreme_points, True)
        self.best_point = memory.best_point.reshape((1, -1))
        self.worst_point = memory.worst_point.reshape((1, -1))
        self.extreme_points = memory.extreme_points
        return chosen