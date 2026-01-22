from abc import ABCMeta, abstractmethod
import sys
class Awaitable(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def __await__(self):
        yield

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Awaitable:
            return _check_methods(C, '__await__')
        return NotImplemented
    __class_getitem__ = classmethod(GenericAlias)