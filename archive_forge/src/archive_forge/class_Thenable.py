import abc
from collections.abc import Callable
class Thenable(Callable, metaclass=abc.ABCMeta):
    """Object that supports ``.then()``."""
    __slots__ = ()

    @abc.abstractmethod
    def then(self, on_success, on_error=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def throw(self, exc=None, tb=None, propagate=True):
        raise NotImplementedError()

    @abc.abstractmethod
    def cancel(self):
        raise NotImplementedError()

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Thenable:
            if any(('then' in B.__dict__ for B in C.__mro__)):
                return True
        return NotImplemented

    @classmethod
    def register(cls, other):
        type(cls).register(cls, other)
        return other