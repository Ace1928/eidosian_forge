from collections import namedtuple, Counter
from os.path import commonprefix
def _count_diff_hashable(actual, expected):
    """Returns list of (cnt_act, cnt_exp, elem) triples where the counts differ"""
    s, t = (Counter(actual), Counter(expected))
    result = []
    for elem, cnt_s in s.items():
        cnt_t = t.get(elem, 0)
        if cnt_s != cnt_t:
            diff = _Mismatch(cnt_s, cnt_t, elem)
            result.append(diff)
    for elem, cnt_t in t.items():
        if elem not in s:
            diff = _Mismatch(0, cnt_t, elem)
            result.append(diff)
    return result