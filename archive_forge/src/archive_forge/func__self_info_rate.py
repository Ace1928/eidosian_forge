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
def _self_info_rate(source):
    """
    returns 'rate of self-information' --
    i.e. average (per-symbol) entropy of the sequence **source**,
    where probability of a given symbol occurring is calculated based on
    the number of occurrences within the sequence itself.

    if all elements of the source are unique, this should equal ``log(len(source), 2)``.

    :arg source:
        iterable containing 0+ symbols
        (e.g. list of strings or ints, string of characters, etc).

    :returns:
        float bits of entropy
    """
    try:
        size = len(source)
    except TypeError:
        size = None
    counts = defaultdict(int)
    for char in source:
        counts[char] += 1
    if size is None:
        values = counts.values()
        size = sum(values)
    else:
        values = itervalues(counts)
    if not size:
        return 0
    return logf(size, 2) - sum((value * logf(value, 2) for value in values)) / size