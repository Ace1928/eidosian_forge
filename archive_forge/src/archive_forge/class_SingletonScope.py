import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
class SingletonScope(Scope):
    """A :class:`Scope` that returns a per-Injector instance for a key.

    :data:`singleton` can be used as a convenience class decorator.

    >>> class A: pass
    >>> injector = Injector()
    >>> provider = ClassProvider(A)
    >>> singleton = SingletonScope(injector)
    >>> a = singleton.get(A, provider)
    >>> b = singleton.get(A, provider)
    >>> a is b
    True
    """
    _context: Dict[type, Provider]

    def configure(self) -> None:
        self._context = {}

    @synchronized(lock)
    def get(self, key: Type[T], provider: Provider[T]) -> Provider[T]:
        try:
            return self._context[key]
        except KeyError:
            instance = self._get_instance(key, provider, self.injector)
            provider = InstanceProvider(instance)
            self._context[key] = provider
            return provider

    def _get_instance(self, key: Type[T], provider: Provider[T], injector: 'Injector') -> T:
        if injector.parent and (not injector.binder.has_explicit_binding_for(key)):
            try:
                return self._get_instance_from_parent(key, provider, injector.parent)
            except (CallError, UnsatisfiedRequirement):
                pass
        return provider.get(injector)

    def _get_instance_from_parent(self, key: Type[T], provider: Provider[T], parent: 'Injector') -> T:
        singleton_scope_binding, _ = parent.binder.get_binding(type(self))
        singleton_scope = singleton_scope_binding.provider.get(parent)
        provider = singleton_scope.get(key, provider)
        return provider.get(parent)