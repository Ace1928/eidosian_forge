import logging
from itertools import groupby
from operator import itemgetter
from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist
def Ae_kappa(self, cA, cB):
    Ae = 0.0
    nitems = float(len(self.I))
    label_freqs = ConditionalFreqDist(((x['labels'], x['coder']) for x in self.data))
    for k in label_freqs.conditions():
        Ae += label_freqs[k][cA] / nitems * (label_freqs[k][cB] / nitems)
    return Ae