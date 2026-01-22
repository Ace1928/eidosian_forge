from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import CLIFilter, to_cli_filter, Never
from prompt_toolkit.keys import Key, Keys
from six import text_type, with_metaclass
class MergedRegistry(_AddRemoveMixin):
    """
    Merge multiple registries of key bindings into one.

    This class acts as a proxy to multiple `Registry` objects, but behaves as
    if this is just one bigger `Registry`.

    :param registries: List of `Registry` objects.
    """

    def __init__(self, registries):
        assert all((isinstance(r, BaseRegistry) for r in registries))
        _AddRemoveMixin.__init__(self)
        self.registries = registries

    def _update_cache(self):
        """
        If one of the original registries was changed. Update our merged
        version.
        """
        expected_version = tuple((r._version for r in self.registries)) + (self._extra_registry._version,)
        if self._last_version != expected_version:
            registry2 = Registry()
            for reg in self.registries:
                registry2.key_bindings.extend(reg.key_bindings)
            registry2.key_bindings.extend(self._extra_registry.key_bindings)
            self._registry2 = registry2
            self._last_version = expected_version