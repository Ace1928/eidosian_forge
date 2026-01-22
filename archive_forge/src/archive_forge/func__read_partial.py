import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _read_partial(self, s, position, reentrances, fstruct=None):
    if fstruct is None:
        if self._START_FDICT_RE.match(s, position):
            fstruct = self._fdict_class()
        else:
            fstruct = self._flist_class()
    match = self._START_FSTRUCT_RE.match(s, position)
    if not match:
        match = self._BARE_PREFIX_RE.match(s, position)
        if not match:
            raise ValueError('open bracket or identifier', position)
    position = match.end()
    if match.group(1):
        identifier = match.group(1)
        if identifier in reentrances:
            raise ValueError('new identifier', match.start(1))
        reentrances[identifier] = fstruct
    if isinstance(fstruct, FeatDict):
        fstruct.clear()
        return self._read_partial_featdict(s, position, match, reentrances, fstruct)
    else:
        del fstruct[:]
        return self._read_partial_featlist(s, position, match, reentrances, fstruct)