import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _read_partial_featlist(self, s, position, match, reentrances, fstruct):
    if match.group(2):
        raise ValueError('open bracket')
    if not match.group(3):
        raise ValueError('open bracket')
    while position < len(s):
        match = self._END_FSTRUCT_RE.match(s, position)
        if match is not None:
            return (fstruct, match.end())
        match = self._REENTRANCE_RE.match(s, position)
        if match:
            position = match.end()
            match = self._TARGET_RE.match(s, position)
            if not match:
                raise ValueError('identifier', position)
            target = match.group(1)
            if target not in reentrances:
                raise ValueError('bound identifier', position)
            position = match.end()
            fstruct.append(reentrances[target])
        else:
            value, position = self._read_value(0, s, position, reentrances)
            fstruct.append(value)
        if self._END_FSTRUCT_RE.match(s, position):
            continue
        match = self._COMMA_RE.match(s, position)
        if match is None:
            raise ValueError('comma', position)
        position = match.end()
    raise ValueError('close bracket', position)