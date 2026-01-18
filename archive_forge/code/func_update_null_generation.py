from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil
def update_null_generation(self, count, alignment_info):
    m = len(alignment_info.trg_sentence) - 1
    fertility_of_null = alignment_info.fertility_of_i(0)
    self.p1 += fertility_of_null * count
    self.p0 += (m - 2 * fertility_of_null) * count