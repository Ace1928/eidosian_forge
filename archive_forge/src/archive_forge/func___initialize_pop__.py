from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np
from ase.db.core import now
from ase.ga import get_raw_score
def __initialize_pop__(self):
    ue = self.use_extinct
    all_cand = self.dc.get_all_relaxed_candidates(use_extinct=ue)
    all_cand.sort(key=lambda x: get_raw_score(x), reverse=True)
    if len(all_cand) > 0:
        fitf = self.__get_fitness__(all_cand)
        all_sorted = list(zip(fitf, all_cand))
        all_sorted.sort(key=itemgetter(0), reverse=True)
        sort_cand = []
        for _, t2 in all_sorted:
            sort_cand.append(t2)
        all_sorted = sort_cand
        i = 0
        while i < len(all_sorted) and len(self.pop) < self.pop_size:
            c = all_sorted[i]
            if self.vf is not None:
                c_vf = self.vf(c)
            i += 1
            eq = False
            for a in self.pop:
                if self.vf is not None:
                    a_vf = self.vf(a)
                    if a_vf == c_vf:
                        if self.comparator.looks_like(a, c):
                            eq = True
                            break
                elif self.comparator.looks_like(a, c):
                    eq = True
                    break
            if not eq:
                self.pop.append(c)
    self.all_cand = all_cand