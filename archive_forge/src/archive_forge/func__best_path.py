from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def _best_path(self, unlabeled_sequence):
    T = len(unlabeled_sequence)
    N = len(self._states)
    self._create_cache()
    self._update_cache(unlabeled_sequence)
    P, O, X, S = self._cache
    V = np.zeros((T, N), np.float32)
    B = -np.ones((T, N), int)
    V[0] = P + O[:, S[unlabeled_sequence[0]]]
    for t in range(1, T):
        for j in range(N):
            vs = V[t - 1, :] + X[:, j]
            best = np.argmax(vs)
            V[t, j] = vs[best] + O[j, S[unlabeled_sequence[t]]]
            B[t, j] = best
    current = np.argmax(V[T - 1, :])
    sequence = [current]
    for t in range(T - 1, 0, -1):
        last = B[t, current]
        sequence.append(last)
        current = last
    sequence.reverse()
    return list(map(self._states.__getitem__, sequence))