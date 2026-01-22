import logging
from itertools import groupby
from operator import itemgetter
from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist
def Do_Kw_pairwise(self, cA, cB, max_distance=1.0):
    """The observed disagreement for the weighted kappa coefficient."""
    total = 0.0
    data = (x for x in self.data if x['coder'] in (cA, cB))
    for i, itemdata in self._grouped_data('item', data):
        total += self.distance(next(itemdata)['labels'], next(itemdata)['labels'])
    ret = total / (len(self.I) * max_distance)
    log.debug('Observed disagreement between %s and %s: %f', cA, cB, ret)
    return ret