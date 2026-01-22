import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
class FeatList(FeatStruct, list):
    """
    A list of feature values, where each feature value is either a
    basic value (such as a string or an integer), or a nested feature
    structure.

    Feature lists may contain reentrant feature values.  A "reentrant
    feature value" is a single feature value that can be accessed via
    multiple feature paths.  Feature lists may also be cyclic.

    Two feature lists are considered equal if they assign the same
    values to all features, and have the same reentrances.

    :see: ``FeatStruct`` for information about feature paths, reentrance,
        cyclic feature structures, mutability, freezing, and hashing.
    """

    def __init__(self, features=()):
        """
        Create a new feature list, with the specified features.

        :param features: The initial list of features for this feature
            list.  If ``features`` is a string, then it is paresd using
            ``FeatStructReader``.  Otherwise, it should be a sequence
            of basic values and nested feature structures.
        """
        if isinstance(features, str):
            FeatStructReader().fromstring(features, self)
        else:
            list.__init__(self, features)
    _INDEX_ERROR = 'Expected int or feature path.  Got %r.'

    def __getitem__(self, name_or_path):
        if isinstance(name_or_path, int):
            return list.__getitem__(self, name_or_path)
        elif isinstance(name_or_path, tuple):
            try:
                val = self
                for fid in name_or_path:
                    if not isinstance(val, FeatStruct):
                        raise KeyError
                    val = val[fid]
                return val
            except (KeyError, IndexError) as e:
                raise KeyError(name_or_path) from e
        else:
            raise TypeError(self._INDEX_ERROR % name_or_path)

    def __delitem__(self, name_or_path):
        """If the feature with the given name or path exists, delete
        its value; otherwise, raise ``KeyError``."""
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        if isinstance(name_or_path, (int, slice)):
            return list.__delitem__(self, name_or_path)
        elif isinstance(name_or_path, tuple):
            if len(name_or_path) == 0:
                raise ValueError('The path () can not be set')
            else:
                parent = self[name_or_path[:-1]]
                if not isinstance(parent, FeatStruct):
                    raise KeyError(name_or_path)
                del parent[name_or_path[-1]]
        else:
            raise TypeError(self._INDEX_ERROR % name_or_path)

    def __setitem__(self, name_or_path, value):
        """Set the value for the feature with the given name or path
        to ``value``.  If ``name_or_path`` is an invalid path, raise
        ``KeyError``."""
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        if isinstance(name_or_path, (int, slice)):
            return list.__setitem__(self, name_or_path, value)
        elif isinstance(name_or_path, tuple):
            if len(name_or_path) == 0:
                raise ValueError('The path () can not be set')
            else:
                parent = self[name_or_path[:-1]]
                if not isinstance(parent, FeatStruct):
                    raise KeyError(name_or_path)
                parent[name_or_path[-1]] = value
        else:
            raise TypeError(self._INDEX_ERROR % name_or_path)
    __iadd__ = _check_frozen(list.__iadd__)
    __imul__ = _check_frozen(list.__imul__)
    append = _check_frozen(list.append)
    extend = _check_frozen(list.extend)
    insert = _check_frozen(list.insert)
    pop = _check_frozen(list.pop)
    remove = _check_frozen(list.remove)
    reverse = _check_frozen(list.reverse)
    sort = _check_frozen(list.sort)

    def __deepcopy__(self, memo):
        memo[id(self)] = selfcopy = self.__class__()
        selfcopy.extend((copy.deepcopy(fval, memo) for fval in self))
        return selfcopy

    def _keys(self):
        return list(range(len(self)))

    def _values(self):
        return self

    def _items(self):
        return enumerate(self)

    def _repr(self, reentrances, reentrance_ids):
        if reentrances[id(self)]:
            assert id(self) not in reentrance_ids
            reentrance_ids[id(self)] = repr(len(reentrance_ids) + 1)
            prefix = '(%s)' % reentrance_ids[id(self)]
        else:
            prefix = ''
        segments = []
        for fval in self:
            if id(fval) in reentrance_ids:
                segments.append('->(%s)' % reentrance_ids[id(fval)])
            elif isinstance(fval, Variable):
                segments.append(fval.name)
            elif isinstance(fval, Expression):
                segments.append('%s' % fval)
            elif isinstance(fval, FeatStruct):
                segments.append(fval._repr(reentrances, reentrance_ids))
            else:
                segments.append('%s' % repr(fval))
        return '{}[{}]'.format(prefix, ', '.join(segments))