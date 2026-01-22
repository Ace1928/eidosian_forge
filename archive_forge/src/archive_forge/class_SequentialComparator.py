import numpy as np
from ase.ga import get_raw_score
class SequentialComparator:
    """Use more than one comparison class and test them all in sequence.

    Supply a list of integers if for example two comparison tests both
    need to be positive if two atoms objects are truly equal.
    Ex:
    methods = [a, b, c, d], logics = [0, 1, 1, 2]
    if a or d is positive -> return True
    if b and c are positive -> return True
    if b and not c are positive (or vice versa) -> return False
    """

    def __init__(self, methods, logics=None):
        if not isinstance(methods, list):
            methods = [methods]
        if logics is None:
            logics = [i for i in range(len(methods))]
        if not isinstance(logics, list):
            logics = [logics]
        assert len(logics) == len(methods)
        self.methods = []
        self.logics = []
        for m, l in zip(methods, logics):
            if hasattr(m, 'looks_like'):
                self.methods.append(m)
                self.logics.append(l)

    def looks_like(self, a1, a2):
        mdct = dict(((l, []) for l in self.logics))
        for m, l in zip(self.methods, self.logics):
            mdct[l].append(m)
        for methods in mdct.values():
            for m in methods:
                if not m.looks_like(a1, a2):
                    break
            else:
                return True
        return False