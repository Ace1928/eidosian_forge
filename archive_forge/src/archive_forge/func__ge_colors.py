import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@property
def _ge_colors(self):
    if self._ge_colors_ is None:
        self._ge_colors_ = partition_to_color(self._ge_partitions)
    return self._ge_colors_