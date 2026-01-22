import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
class FeatDict(FeatStruct, dict):
    """
    A feature structure that acts like a Python dictionary.  I.e., a
    mapping from feature identifiers to feature values, where a feature
    identifier can be a string or a ``Feature``; and where a feature value
    can be either a basic value (such as a string or an integer), or a nested
    feature structure.  A feature identifiers for a ``FeatDict`` is
    sometimes called a "feature name".

    Two feature dicts are considered equal if they assign the same
    values to all features, and have the same reentrances.

    :see: ``FeatStruct`` for information about feature paths, reentrance,
        cyclic feature structures, mutability, freezing, and hashing.
    """

    def __init__(self, features=None, **morefeatures):
        """
        Create a new feature dictionary, with the specified features.

        :param features: The initial value for this feature
            dictionary.  If ``features`` is a ``FeatStruct``, then its
            features are copied (shallow copy).  If ``features`` is a
            dict, then a feature is created for each item, mapping its
            key to its value.  If ``features`` is a string, then it is
            processed using ``FeatStructReader``.  If ``features`` is a list of
            tuples ``(name, val)``, then a feature is created for each tuple.
        :param morefeatures: Additional features for the new feature
            dictionary.  If a feature is listed under both ``features`` and
            ``morefeatures``, then the value from ``morefeatures`` will be
            used.
        """
        if isinstance(features, str):
            FeatStructReader().fromstring(features, self)
            self.update(**morefeatures)
        else:
            self.update(features, **morefeatures)
    _INDEX_ERROR = 'Expected feature name or path.  Got %r.'

    def __getitem__(self, name_or_path):
        """If the feature with the given name or path exists, return
        its value; otherwise, raise ``KeyError``."""
        if isinstance(name_or_path, (str, Feature)):
            return dict.__getitem__(self, name_or_path)
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

    def get(self, name_or_path, default=None):
        """If the feature with the given name or path exists, return its
        value; otherwise, return ``default``."""
        try:
            return self[name_or_path]
        except KeyError:
            return default

    def __contains__(self, name_or_path):
        """Return true if a feature with the given name or path exists."""
        try:
            self[name_or_path]
            return True
        except KeyError:
            return False

    def has_key(self, name_or_path):
        """Return true if a feature with the given name or path exists."""
        return name_or_path in self

    def __delitem__(self, name_or_path):
        """If the feature with the given name or path exists, delete
        its value; otherwise, raise ``KeyError``."""
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        if isinstance(name_or_path, (str, Feature)):
            return dict.__delitem__(self, name_or_path)
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
        if isinstance(name_or_path, (str, Feature)):
            return dict.__setitem__(self, name_or_path, value)
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
    clear = _check_frozen(dict.clear)
    pop = _check_frozen(dict.pop)
    popitem = _check_frozen(dict.popitem)
    setdefault = _check_frozen(dict.setdefault)

    def update(self, features=None, **morefeatures):
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        if features is None:
            items = ()
        elif hasattr(features, 'items') and callable(features.items):
            items = features.items()
        elif hasattr(features, '__iter__'):
            items = features
        else:
            raise ValueError('Expected mapping or list of tuples')
        for key, val in items:
            if not isinstance(key, (str, Feature)):
                raise TypeError('Feature names must be strings')
            self[key] = val
        for key, val in morefeatures.items():
            if not isinstance(key, (str, Feature)):
                raise TypeError('Feature names must be strings')
            self[key] = val

    def __deepcopy__(self, memo):
        memo[id(self)] = selfcopy = self.__class__()
        for key, val in self._items():
            selfcopy[copy.deepcopy(key, memo)] = copy.deepcopy(val, memo)
        return selfcopy

    def _keys(self):
        return self.keys()

    def _values(self):
        return self.values()

    def _items(self):
        return self.items()

    def __str__(self):
        """
        Display a multi-line representation of this feature dictionary
        as an FVM (feature value matrix).
        """
        return '\n'.join(self._str(self._find_reentrances({}), {}))

    def _repr(self, reentrances, reentrance_ids):
        segments = []
        prefix = ''
        suffix = ''
        if reentrances[id(self)]:
            assert id(self) not in reentrance_ids
            reentrance_ids[id(self)] = repr(len(reentrance_ids) + 1)
        for fname, fval in sorted(self.items()):
            display = getattr(fname, 'display', None)
            if id(fval) in reentrance_ids:
                segments.append(f'{fname}->({reentrance_ids[id(fval)]})')
            elif display == 'prefix' and (not prefix) and isinstance(fval, (Variable, str)):
                prefix = '%s' % fval
            elif display == 'slash' and (not suffix):
                if isinstance(fval, Variable):
                    suffix = '/%s' % fval.name
                else:
                    suffix = '/%s' % repr(fval)
            elif isinstance(fval, Variable):
                segments.append(f'{fname}={fval.name}')
            elif fval is True:
                segments.append('+%s' % fname)
            elif fval is False:
                segments.append('-%s' % fname)
            elif isinstance(fval, Expression):
                segments.append(f'{fname}=<{fval}>')
            elif not isinstance(fval, FeatStruct):
                segments.append(f'{fname}={repr(fval)}')
            else:
                fval_repr = fval._repr(reentrances, reentrance_ids)
                segments.append(f'{fname}={fval_repr}')
        if reentrances[id(self)]:
            prefix = f'({reentrance_ids[id(self)]}){prefix}'
        return '{}[{}]{}'.format(prefix, ', '.join(segments), suffix)

    def _str(self, reentrances, reentrance_ids):
        """
        :return: A list of lines composing a string representation of
            this feature dictionary.
        :param reentrances: A dictionary that maps from the ``id`` of
            each feature value in self, indicating whether that value
            is reentrant or not.
        :param reentrance_ids: A dictionary mapping from each ``id``
            of a feature value to a unique identifier.  This is modified
            by ``repr``: the first time a reentrant feature value is
            displayed, an identifier is added to ``reentrance_ids`` for
            it.
        """
        if reentrances[id(self)]:
            assert id(self) not in reentrance_ids
            reentrance_ids[id(self)] = repr(len(reentrance_ids) + 1)
        if len(self) == 0:
            if reentrances[id(self)]:
                return ['(%s) []' % reentrance_ids[id(self)]]
            else:
                return ['[]']
        maxfnamelen = max((len('%s' % k) for k in self.keys()))
        lines = []
        for fname, fval in sorted(self.items()):
            fname = ('%s' % fname).ljust(maxfnamelen)
            if isinstance(fval, Variable):
                lines.append(f'{fname} = {fval.name}')
            elif isinstance(fval, Expression):
                lines.append(f'{fname} = <{fval}>')
            elif isinstance(fval, FeatList):
                fval_repr = fval._repr(reentrances, reentrance_ids)
                lines.append(f'{fname} = {repr(fval_repr)}')
            elif not isinstance(fval, FeatDict):
                lines.append(f'{fname} = {repr(fval)}')
            elif id(fval) in reentrance_ids:
                lines.append(f'{fname} -> ({reentrance_ids[id(fval)]})')
            else:
                if lines and lines[-1] != '':
                    lines.append('')
                fval_lines = fval._str(reentrances, reentrance_ids)
                fval_lines = [' ' * (maxfnamelen + 3) + l for l in fval_lines]
                nameline = (len(fval_lines) - 1) // 2
                fval_lines[nameline] = fname + ' =' + fval_lines[nameline][maxfnamelen + 2:]
                lines += fval_lines
                lines.append('')
        if lines[-1] == '':
            lines.pop()
        maxlen = max((len(line) for line in lines))
        lines = ['[ {}{} ]'.format(line, ' ' * (maxlen - len(line))) for line in lines]
        if reentrances[id(self)]:
            idstr = '(%s) ' % reentrance_ids[id(self)]
            lines = [' ' * len(idstr) + l for l in lines]
            idline = (len(lines) - 1) // 2
            lines[idline] = idstr + lines[idline][len(idstr):]
        return lines