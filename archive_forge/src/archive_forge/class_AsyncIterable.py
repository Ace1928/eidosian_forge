from abc import ABCMeta, abstractmethod
import sys
class AsyncIterable(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def __aiter__(self):
        return AsyncIterator()

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AsyncIterable:
            return _check_methods(C, '__aiter__')
        return NotImplemented
    __class_getitem__ = classmethod(GenericAlias)