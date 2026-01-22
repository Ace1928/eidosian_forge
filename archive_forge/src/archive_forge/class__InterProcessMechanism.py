from abc import ABC
from abc import abstractmethod
import errno
import os
class _InterProcessMechanism(ABC):

    @staticmethod
    @abstractmethod
    def trylock(lockfile):
        ...

    @staticmethod
    @abstractmethod
    def unlock(lockfile):
        ...