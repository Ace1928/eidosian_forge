import abc
import collections
import collections.abc
import operator
import sys
import typing
class AsyncContextManager(typing.Generic[T_co]):
    __slots__ = ()

    async def __aenter__(self):
        return self

    @abc.abstractmethod
    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AsyncContextManager:
            return _check_methods_in_mro(C, '__aenter__', '__aexit__')
        return NotImplemented