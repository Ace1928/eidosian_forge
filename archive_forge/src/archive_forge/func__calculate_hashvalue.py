import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _calculate_hashvalue(self, visited):
    """
        Return a hash value for this feature structure.

        :require: ``self`` must be frozen.
        :param visited: A set containing the ids of all feature
            structures we've already visited while hashing.
        """
    if id(self) in visited:
        return 1
    visited.add(id(self))
    hashval = 5831
    for fname, fval in sorted(self._items()):
        hashval *= 37
        hashval += hash(fname)
        hashval *= 37
        if isinstance(fval, FeatStruct):
            hashval += fval._calculate_hashvalue(visited)
        else:
            hashval += hash(fval)
        hashval = int(hashval & 2147483647)
    return hashval