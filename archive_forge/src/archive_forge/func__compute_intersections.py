import os
from random import choice
from itertools import combinations
import snappy
from plink import LinkManager
from .twister_core import build_bundle, build_splitting, twister_version
def _compute_intersections(curves):
    curve_names = [name for name, inverse_name, data in curves['annulus']] + [name for name, inverse_name, data in curves['rectangle']]
    data = dict(((name, set([abs(int(x)) for x in data])) for name, inverse_name, data in curves['annulus'] + curves['rectangle']))
    rows = [[''] + curve_names]
    for name in curve_names:
        row = [name]
        for name2 in curve_names:
            row.append(len(data[name].intersection(data[name2])) if name != name2 else 0)
        rows.append(row)
    return rows