import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _read_partial_featdict(self, s, position, match, reentrances, fstruct):
    if match.group(2):
        if self._prefix_feature is None:
            raise ValueError('open bracket or identifier', match.start(2))
        prefixval = match.group(2).strip()
        if prefixval.startswith('?'):
            prefixval = Variable(prefixval)
        fstruct[self._prefix_feature] = prefixval
    if not match.group(3):
        return self._finalize(s, match.end(), reentrances, fstruct)
    while position < len(s):
        name = value = None
        match = self._END_FSTRUCT_RE.match(s, position)
        if match is not None:
            return self._finalize(s, match.end(), reentrances, fstruct)
        match = self._FEATURE_NAME_RE.match(s, position)
        if match is None:
            raise ValueError('feature name', position)
        name = match.group(2)
        position = match.end()
        if name[0] == '*' and name[-1] == '*':
            name = self._features.get(name[1:-1])
            if name is None:
                raise ValueError('known special feature', match.start(2))
        if name in fstruct:
            raise ValueError('new name', match.start(2))
        if match.group(1) == '+':
            value = True
        if match.group(1) == '-':
            value = False
        if value is None:
            match = self._REENTRANCE_RE.match(s, position)
            if match is not None:
                position = match.end()
                match = self._TARGET_RE.match(s, position)
                if not match:
                    raise ValueError('identifier', position)
                target = match.group(1)
                if target not in reentrances:
                    raise ValueError('bound identifier', position)
                position = match.end()
                value = reentrances[target]
        if value is None:
            match = self._ASSIGN_RE.match(s, position)
            if match:
                position = match.end()
                value, position = self._read_value(name, s, position, reentrances)
            else:
                raise ValueError('equals sign', position)
        fstruct[name] = value
        if self._END_FSTRUCT_RE.match(s, position):
            continue
        match = self._COMMA_RE.match(s, position)
        if match is None:
            raise ValueError('comma', position)
        position = match.end()
    raise ValueError('close bracket', position)