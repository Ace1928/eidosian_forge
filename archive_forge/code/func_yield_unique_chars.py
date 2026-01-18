import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def yield_unique_chars(chars: list):
    sorted_chars = sorted(chars, key=key)
    for grp, grp_chars in itertools.groupby(sorted_chars, key=key):
        for y_cluster in cluster_objects(list(grp_chars), itemgetter('doctop'), tolerance):
            for x_cluster in cluster_objects(y_cluster, itemgetter('x0'), tolerance):
                yield sorted(x_cluster, key=pos_key)[0]