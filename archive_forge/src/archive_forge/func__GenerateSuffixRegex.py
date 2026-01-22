from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import math
import re
import six
def _GenerateSuffixRegex():
    """Creates a suffix regex for human-readable byte counts."""
    human_bytes_re = '(?P<num>\\d*\\.\\d+|\\d+)\\s*(?P<suffix>%s)?'
    suffixes = []
    suffix_to_si = {}
    for i, si in enumerate(_EXP_STRINGS):
        si_suffixes = [s.lower() for s in list(si)[1:]]
        for suffix in si_suffixes:
            suffix_to_si[suffix] = i
        suffixes.extend(si_suffixes)
    human_bytes_re %= '|'.join(suffixes)
    matcher = re.compile(human_bytes_re)
    return (suffix_to_si, matcher)