import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@property
def _sge_colors(self):
    if self._sge_colors_ is None:
        self._sge_colors_ = partition_to_color(self._sge_partitions)
    return self._sge_colors_