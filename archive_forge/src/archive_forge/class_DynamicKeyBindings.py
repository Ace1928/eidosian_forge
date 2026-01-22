from __future__ import annotations
from abc import ABCMeta, abstractmethod, abstractproperty
from inspect import isawaitable
from typing import (
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import FilterOrBool, Never, to_filter
from prompt_toolkit.keys import KEY_ALIASES, Keys
class DynamicKeyBindings(_Proxy):
    """
    KeyBindings class that can dynamically returns any KeyBindings.

    :param get_key_bindings: Callable that returns a :class:`.KeyBindings` instance.
    """

    def __init__(self, get_key_bindings: Callable[[], KeyBindingsBase | None]) -> None:
        self.get_key_bindings = get_key_bindings
        self.__version = 0
        self._last_child_version = None
        self._dummy = KeyBindings()

    def _update_cache(self) -> None:
        key_bindings = self.get_key_bindings() or self._dummy
        assert isinstance(key_bindings, KeyBindingsBase)
        version = (id(key_bindings), key_bindings._version)
        self._bindings2 = key_bindings
        self._last_version = version