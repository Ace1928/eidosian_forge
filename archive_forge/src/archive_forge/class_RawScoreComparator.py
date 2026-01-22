import numpy as np
from ase.ga import get_raw_score
class RawScoreComparator:
    """Compares the raw_score of the supplied individuals
       objects using a1.info['key_value_pairs']['raw_score'].

       Parameters:

       dist: the difference in raw_score below which two
       scores are deemed equal.
    """

    def __init__(self, dist=0.02):
        self.dist = dist

    def looks_like(self, a1, a2):
        d = abs(get_raw_score(a1) - get_raw_score(a2))
        if d >= self.dist:
            return False
        else:
            return True