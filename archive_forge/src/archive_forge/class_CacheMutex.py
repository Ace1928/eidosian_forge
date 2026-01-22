from __future__ import annotations
import abc
import pickle
import time
from typing import Any
from typing import Callable
from typing import cast
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Union
from ..util.typing import Self
class CacheMutex(abc.ABC):
    """Describes a mutexing object with acquire and release methods.

    This is an abstract base class; any object that has acquire/release
    methods may be used.

    .. versionadded:: 1.1


    .. seealso::

        :meth:`.CacheBackend.get_mutex` - the backend method that optionally
        returns this locking object.

    """

    @abc.abstractmethod
    def acquire(self, wait: bool=True) -> bool:
        """Acquire the mutex.

        :param wait: if True, block until available, else return True/False
         immediately.

        :return: True if the lock succeeded.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def release(self) -> None:
        """Release the mutex."""
        raise NotImplementedError()

    @abc.abstractmethod
    def locked(self) -> bool:
        """Check if the mutex was acquired.

        :return: true if the lock is acquired.

        .. versionadded:: 1.1.2

        """
        raise NotImplementedError()

    @classmethod
    def __subclasshook__(cls, C):
        return hasattr(C, 'acquire') and hasattr(C, 'release')