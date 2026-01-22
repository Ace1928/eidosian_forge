from math import log
from operator import itemgetter
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.tag.api import TaggerI
def _safe_div(self, v1, v2):
    """
        Safe floating point division function, does not allow division by 0
        returns -1 if the denominator is 0
        """
    if v2 == 0:
        return -1
    else:
        return v1 / v2