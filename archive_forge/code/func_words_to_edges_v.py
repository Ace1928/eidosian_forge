import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def words_to_edges_v(words, word_threshold: int=DEFAULT_MIN_WORDS_VERTICAL):
    """
    Find (imaginary) vertical lines that connect the left, right, or
    center of at least `word_threshold` words.
    """
    by_x0 = cluster_objects(words, itemgetter('x0'), 1)
    by_x1 = cluster_objects(words, itemgetter('x1'), 1)

    def get_center(word):
        return float(word['x0'] + word['x1']) / 2
    by_center = cluster_objects(words, get_center, 1)
    clusters = by_x0 + by_x1 + by_center
    sorted_clusters = sorted(clusters, key=lambda x: -len(x))
    large_clusters = filter(lambda x: len(x) >= word_threshold, sorted_clusters)
    bboxes = list(map(objects_to_bbox, large_clusters))
    condensed_bboxes = []
    for bbox in bboxes:
        overlap = any((get_bbox_overlap(bbox, c) for c in condensed_bboxes))
        if not overlap:
            condensed_bboxes.append(bbox)
    if len(condensed_bboxes) == 0:
        return []
    condensed_rects = map(bbox_to_rect, condensed_bboxes)
    sorted_rects = list(sorted(condensed_rects, key=itemgetter('x0')))
    max_x1 = max(map(itemgetter('x1'), sorted_rects))
    min_top = min(map(itemgetter('top'), sorted_rects))
    max_bottom = max(map(itemgetter('bottom'), sorted_rects))
    return [{'x0': b['x0'], 'x1': b['x0'], 'top': min_top, 'bottom': max_bottom, 'height': max_bottom - min_top, 'orientation': 'v'} for b in sorted_rects] + [{'x0': max_x1, 'x1': max_x1, 'top': min_top, 'bottom': max_bottom, 'height': max_bottom - min_top, 'orientation': 'v'}]