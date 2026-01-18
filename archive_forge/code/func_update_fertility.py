from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil
def update_fertility(self, count, alignment_info):
    for i in range(0, len(alignment_info.src_sentence)):
        s = alignment_info.src_sentence[i]
        phi = alignment_info.fertility_of_i(i)
        self.fertility[phi][s] += count
        self.fertility_for_any_phi[s] += count