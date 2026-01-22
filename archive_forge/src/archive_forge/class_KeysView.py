from abc import ABCMeta, abstractmethod
import sys
class KeysView(MappingView, Set):
    __slots__ = ()

    @classmethod
    def _from_iterable(cls, it):
        return set(it)

    def __contains__(self, key):
        return key in self._mapping

    def __iter__(self):
        yield from self._mapping