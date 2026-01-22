from abc import ABCMeta, abstractmethod
from collections.abc import Callable
class CallableSignature(CallableTask):
    """Celery Signature interface."""
    __required_attributes__ = frozenset({'clone', 'freeze', 'set', 'link', 'link_error', '__or__'})

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def type(self):
        pass

    @property
    @abstractmethod
    def app(self):
        pass

    @property
    @abstractmethod
    def id(self):
        pass

    @property
    @abstractmethod
    def task(self):
        pass

    @property
    @abstractmethod
    def args(self):
        pass

    @property
    @abstractmethod
    def kwargs(self):
        pass

    @property
    @abstractmethod
    def options(self):
        pass

    @property
    @abstractmethod
    def subtask_type(self):
        pass

    @property
    @abstractmethod
    def chord_size(self):
        pass

    @property
    @abstractmethod
    def immutable(self):
        pass

    @abstractmethod
    def clone(self, args=None, kwargs=None):
        pass

    @abstractmethod
    def freeze(self, id=None, group_id=None, chord=None, root_id=None, group_index=None):
        pass

    @abstractmethod
    def set(self, immutable=None, **options):
        pass

    @abstractmethod
    def link(self, callback):
        pass

    @abstractmethod
    def link_error(self, errback):
        pass

    @abstractmethod
    def __or__(self, other):
        pass

    @abstractmethod
    def __invert__(self):
        pass

    @classmethod
    def __subclasshook__(cls, C):
        return cls._subclasshook_using(CallableSignature, C)