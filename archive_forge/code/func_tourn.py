import bisect
from collections import defaultdict, namedtuple
from itertools import chain
import math
from operator import attrgetter, itemgetter
import random
import numpy
def tourn(ind1, ind2):
    if ind1.fitness.dominates(ind2.fitness):
        return ind1
    elif ind2.fitness.dominates(ind1.fitness):
        return ind2
    if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
        return ind2
    elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
        return ind1
    if random.random() <= 0.5:
        return ind1
    return ind2