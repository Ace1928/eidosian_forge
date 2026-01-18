from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil
def zero_indexed_alignment(self):
    """
        :return: Zero-indexed alignment, suitable for use in external
            ``nltk.translate`` modules like ``nltk.translate.Alignment``
        :rtype: list(tuple)
        """
    zero_indexed_alignment = []
    for j in range(1, len(self.trg_sentence)):
        i = self.alignment[j] - 1
        if i < 0:
            i = None
        zero_indexed_alignment.append((j - 1, i))
    return zero_indexed_alignment