from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import CLIFilter, to_cli_filter, Never
from prompt_toolkit.keys import Key, Keys
from six import text_type, with_metaclass
class _AddRemoveMixin(BaseRegistry):
    """
    Common part for ConditionalRegistry and MergedRegistry.
    """

    def __init__(self):
        self._registry2 = Registry()
        self._last_version = None
        self._extra_registry = Registry()

    def _update_cache(self):
        raise NotImplementedError

    def add_binding(self, *k, **kw):
        return self._extra_registry.add_binding(*k, **kw)

    def remove_binding(self, *k, **kw):
        return self._extra_registry.remove_binding(*k, **kw)

    @property
    def key_bindings(self):
        self._update_cache()
        return self._registry2.key_bindings

    @property
    def _version(self):
        self._update_cache()
        return self._last_version

    def get_bindings_for_keys(self, *a, **kw):
        self._update_cache()
        return self._registry2.get_bindings_for_keys(*a, **kw)

    def get_bindings_starting_with_keys(self, *a, **kw):
        self._update_cache()
        return self._registry2.get_bindings_starting_with_keys(*a, **kw)