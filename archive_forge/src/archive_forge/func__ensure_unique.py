from __future__ import absolute_import, division, print_function, unicode_literals
import codecs
from collections import defaultdict
from math import ceil, log as logf
import logging; log = logging.getLogger(__name__)
import pkg_resources
import os
from passlib import exc
from passlib.utils.compat import PY2, irange, itervalues, int_types
from passlib.utils import rng, getrandstr, to_unicode
from passlib.utils.decor import memoized_property
def _ensure_unique(source, param='source'):
    """
    helper for generators --
    Throws ValueError if source elements aren't unique.
    Error message will display (abbreviated) repr of the duplicates in a string/list
    """
    cache = _ensure_unique_cache
    hashable = True
    try:
        if source in cache:
            return True
    except TypeError:
        hashable = False
    if isinstance(source, _set_types) or len(set(source)) == len(source):
        if hashable:
            try:
                cache.add(source)
            except TypeError:
                pass
        return True
    seen = set()
    dups = set()
    for elem in source:
        (dups if elem in seen else seen).add(elem)
    dups = sorted(dups)
    trunc = 8
    if len(dups) > trunc:
        trunc = 5
    dup_repr = ', '.join((repr(str(word)) for word in dups[:trunc]))
    if len(dups) > trunc:
        dup_repr += ', ... plus %d others' % (len(dups) - trunc)
    raise ValueError('`%s` cannot contain duplicate elements: %s' % (param, dup_repr))