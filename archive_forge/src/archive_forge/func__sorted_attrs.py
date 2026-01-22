import collections
import copy
import itertools
import random
import re
import warnings
def _sorted_attrs(elem):
    """Get a flat list of elem's attributes, sorted for consistency (PRIVATE)."""
    singles = []
    lists = []
    for attrname, child in sorted(elem.__dict__.items(), key=lambda kv: kv[0]):
        if child is None:
            continue
        if isinstance(child, list):
            lists.extend(child)
        else:
            singles.append(child)
    return (x for x in singles + lists if isinstance(x, TreeElement))