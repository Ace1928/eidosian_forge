import sys
import types
from array import array
from collections import abc
from ._abc import MultiMapping, MutableMultiMapping
class _ItemsView(_ViewBase, abc.ItemsView):

    def __contains__(self, item):
        assert isinstance(item, tuple) or isinstance(item, list)
        assert len(item) == 2
        for i, k, v in self._impl._items:
            if item[0] == k and item[1] == v:
                return True
        return False

    def __iter__(self):
        return _Iter(len(self), self._iter(self._impl._version))

    def _iter(self, version):
        for i, k, v in self._impl._items:
            if version != self._impl._version:
                raise RuntimeError('Dictionary changed during iteration')
            yield (k, v)

    def __repr__(self):
        lst = []
        for item in self._impl._items:
            lst.append('{!r}: {!r}'.format(item[1], item[2]))
        body = ', '.join(lst)
        return '{}({})'.format(self.__class__.__name__, body)