import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def add_lookaheads(self, lookbacks, followset):
    for trans, lb in lookbacks.items():
        for state, p in lb:
            if state not in p.lookaheads:
                p.lookaheads[state] = []
            f = followset.get(trans, [])
            for a in f:
                if a not in p.lookaheads[state]:
                    p.lookaheads[state].append(a)