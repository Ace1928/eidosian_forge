import logging
import pickle
import random
from collections import defaultdict
from nltk import jsontags
from nltk.data import find, load
from nltk.tag.api import TaggerI
def upd_feat(c, f, w, v):
    param = (f, c)
    self._totals[param] += (self.i - self._tstamps[param]) * w
    self._tstamps[param] = self.i
    self.weights[f][c] = w + v