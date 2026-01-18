import json
import math
import re
import signal
from contextlib import contextmanager
from glob import glob
from os.path import join as pjoin
def tf_idf_vec_uni(sentence, vocounts, totcounts):
    sen_tab = sentence.lower().split()
    uni_dic = {}
    for _, w in enumerate(sen_tab):
        if w in vocounts:
            uni_dic[w] = -math.log(float(vocounts.get(w, 1.0)) / totcounts)
    uni_norm = math.sqrt(sum([x * x for x in uni_dic.values()]))
    if uni_norm > 0:
        for w in uni_dic:
            uni_dic[w] /= uni_norm
    return uni_dic