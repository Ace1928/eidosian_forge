import abc
from typing import Any
class _AbstractProvider(_with_metaclass(abc.ABCMeta)):

    @abc.abstractmethod
    def can_provide(self, type_object, type_name):
        raise NotImplementedError