import logging
from itertools import groupby
from operator import itemgetter
from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist
def weighted_kappa(self, max_distance=1.0):
    """Cohen 1968"""
    return self._pairwise_average(lambda cA, cB: self.weighted_kappa_pairwise(cA, cB, max_distance))