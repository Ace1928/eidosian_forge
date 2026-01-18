import json
import math
import re
import signal
from contextlib import contextmanager
from glob import glob
from os.path import join as pjoin
def word_url_tokenize(st, max_len=20480, max_cont_len=512):
    stp = ' '.join([w[:max_cont_len] if w[:max_cont_len].count('.') <= 12 else '.' for w in st.split()[:max_len]])
    try:
        with time_limit(2):
            return pre_word_url_tokenize(stp)
    except TimeoutException:
        print('timeout', len(st), ' --n-- '.join(st[:128].split('\n')))
        print(' --n-- '.join(stp.split('\n')))
        res = ('missed page', [])
        return res