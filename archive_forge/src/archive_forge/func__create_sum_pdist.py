import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def _create_sum_pdist(numsamples):
    """
    Return the true probability distribution for the experiment
    ``_create_rand_fdist(numsamples, x)``.
    """
    fdist = FreqDist()
    for x in range(1, (1 + numsamples) // 2 + 1):
        for y in range(0, numsamples // 2 + 1):
            fdist[x + y] += 1
    return MLEProbDist(fdist)