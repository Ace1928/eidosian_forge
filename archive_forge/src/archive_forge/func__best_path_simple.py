from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def _best_path_simple(self, unlabeled_sequence):
    T = len(unlabeled_sequence)
    N = len(self._states)
    V = np.zeros((T, N), np.float64)
    B = {}
    symbol = unlabeled_sequence[0]
    for i, state in enumerate(self._states):
        V[0, i] = self._priors.logprob(state) + self._output_logprob(state, symbol)
        B[0, state] = None
    for t in range(1, T):
        symbol = unlabeled_sequence[t]
        for j in range(N):
            sj = self._states[j]
            best = None
            for i in range(N):
                si = self._states[i]
                va = V[t - 1, i] + self._transitions[si].logprob(sj)
                if not best or va > best[0]:
                    best = (va, si)
            V[t, j] = best[0] + self._output_logprob(sj, symbol)
            B[t, sj] = best[1]
    best = None
    for i in range(N):
        val = V[T - 1, i]
        if not best or val > best[0]:
            best = (val, self._states[i])
    current = best[1]
    sequence = [current]
    for t in range(T - 1, 0, -1):
        last = B[t, current]
        sequence.append(last)
        current = last
    sequence.reverse()
    return sequence