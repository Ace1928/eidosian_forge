import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def _r_Nr(self):
    """
        Split the frequency distribution in two list (r, Nr), where Nr(r) > 0
        """
    nonzero = self._r_Nr_non_zero()
    if not nonzero:
        return ([], [])
    return zip(*sorted(nonzero.items()))