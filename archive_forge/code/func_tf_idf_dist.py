import json
import math
import re
import signal
from contextlib import contextmanager
from glob import glob
from os.path import join as pjoin
def tf_idf_dist(dic_q, dic_t):
    dot_p = 0
    if len(dic_t) < len(dic_q):
        for w, x in dic_t.items():
            if w in dic_q:
                dot_p += x * dic_q[w]
    else:
        for w, x in dic_q.items():
            if w in dic_t:
                dot_p += x * dic_t[w]
    return dot_p