import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def words_to_edges_h(words, word_threshold: int=DEFAULT_MIN_WORDS_HORIZONTAL):
    """
    Find (imaginary) horizontal lines that connect the tops
    of at least `word_threshold` words.
    """
    by_top = cluster_objects(words, itemgetter('top'), 1)
    large_clusters = filter(lambda x: len(x) >= word_threshold, by_top)
    rects = list(map(objects_to_rect, large_clusters))
    if len(rects) == 0:
        return []
    min_x0 = min(map(itemgetter('x0'), rects))
    max_x1 = max(map(itemgetter('x1'), rects))
    edges = []
    for r in rects:
        edges += [{'x0': min_x0, 'x1': max_x1, 'top': r['top'], 'bottom': r['top'], 'width': max_x1 - min_x0, 'orientation': 'h'}, {'x0': min_x0, 'x1': max_x1, 'top': r['bottom'], 'bottom': r['bottom'], 'width': max_x1 - min_x0, 'orientation': 'h'}]
    return edges