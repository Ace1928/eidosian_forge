from abc import ABC
from abc import abstractmethod
import errno
import os
class _InterProcessReaderWriterLockMechanism(ABC):

    @staticmethod
    @abstractmethod
    def trylock(lockfile, exclusive):
        ...

    @staticmethod
    @abstractmethod
    def unlock(lockfile):
        ...

    @staticmethod
    @abstractmethod
    def get_handle(path):
        ...

    @staticmethod
    @abstractmethod
    def close_handle(lockfile):
        ...