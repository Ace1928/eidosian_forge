import builtins
import datetime as dt
import re
import weakref
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import partial
from itertools import chain
from operator import itemgetter
import numpy as np
import param
from . import util
from .accessors import Apply, Opts, Redim
from .options import Options, Store, cleanup_custom_options
from .pprint import PrettyPrinter
from .tree import AttrTree
from .util import bytes_to_unicode
@classmethod
def _deduplicate_items(cls, items):
    """Deduplicates assigned paths by incrementing numbering"""
    counter = Counter([path[:i] for path, _ in items for i in range(1, len(path) + 1)])
    if sum(counter.values()) == len(counter):
        return items
    new_items = []
    counts = defaultdict(lambda: 0)
    for path, item in items:
        if counter[path] > 1:
            path = path + (util.int_to_roman(counts[path] + 1),)
        else:
            inc = 1
            while counts[path]:
                path = path[:-1] + (util.int_to_roman(counts[path] + inc),)
                inc += 1
        new_items.append((path, item))
        counts[path] += 1
    return new_items