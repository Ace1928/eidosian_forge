from abc import ABCMeta, abstractmethod
import sys
class Reversible(Iterable):
    __slots__ = ()

    @abstractmethod
    def __reversed__(self):
        while False:
            yield None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Reversible:
            return _check_methods(C, '__reversed__', '__iter__')
        return NotImplemented