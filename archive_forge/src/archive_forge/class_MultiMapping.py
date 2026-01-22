import abc
import sys
import types
from collections.abc import Mapping, MutableMapping
class MultiMapping(Mapping, metaclass=_TypingMeta):

    @abc.abstractmethod
    def getall(self, key, default=None):
        raise KeyError

    @abc.abstractmethod
    def getone(self, key, default=None):
        raise KeyError