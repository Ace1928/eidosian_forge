import logging
from itertools import groupby
from operator import itemgetter
from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist
def Disagreement(self, label_freqs):
    total_labels = sum(label_freqs.values())
    pairs = 0.0
    for j, nj in label_freqs.items():
        for l, nl in label_freqs.items():
            pairs += float(nj * nl) * self.distance(l, j)
    return 1.0 * pairs / (total_labels * (total_labels - 1))