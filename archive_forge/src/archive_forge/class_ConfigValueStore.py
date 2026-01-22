import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
class ConfigValueStore:
    """The ConfigValueStore object stores configuration values."""

    def __init__(self, mapping=None):
        """Initialize a ConfigValueStore.

        :type mapping: dict
        :param mapping: The mapping parameter is a map of string to a subclass
            of BaseProvider. When a config variable is asked for via the
            get_config_variable method, the corresponding provider will be
            invoked to load the value.
        """
        self._overrides = {}
        self._mapping = {}
        if mapping is not None:
            for logical_name, provider in mapping.items():
                self.set_config_provider(logical_name, provider)

    def __deepcopy__(self, memo):
        config_store = ConfigValueStore(copy.deepcopy(self._mapping, memo))
        for logical_name, override_value in self._overrides.items():
            config_store.set_config_variable(logical_name, override_value)
        return config_store

    def __copy__(self):
        config_store = ConfigValueStore(copy.copy(self._mapping))
        for logical_name, override_value in self._overrides.items():
            config_store.set_config_variable(logical_name, override_value)
        return config_store

    def get_config_variable(self, logical_name):
        """
        Retrieve the value associeated with the specified logical_name
        from the corresponding provider. If no value is found None will
        be returned.

        :type logical_name: str
        :param logical_name: The logical name of the session variable
            you want to retrieve.  This name will be mapped to the
            appropriate environment variable name for this session as
            well as the appropriate config file entry.

        :returns: value of variable or None if not defined.
        """
        if logical_name in self._overrides:
            return self._overrides[logical_name]
        if logical_name not in self._mapping:
            return None
        provider = self._mapping[logical_name]
        return provider.provide()

    def get_config_provider(self, logical_name):
        """
        Retrieve the provider associated with the specified logical_name.
        If no provider is found None will be returned.

        :type logical_name: str
        :param logical_name: The logical name of the session variable
            you want to retrieve.  This name will be mapped to the
            appropriate environment variable name for this session as
            well as the appropriate config file entry.

        :returns: configuration provider or None if not defined.
        """
        if logical_name in self._overrides or logical_name not in self._mapping:
            return None
        provider = self._mapping[logical_name]
        return provider

    def set_config_variable(self, logical_name, value):
        """Set a configuration variable to a specific value.

        By using this method, you can override the normal lookup
        process used in ``get_config_variable`` by explicitly setting
        a value.  Subsequent calls to ``get_config_variable`` will
        use the ``value``.  This gives you per-session specific
        configuration values.

        ::
            >>> # Assume logical name 'foo' maps to env var 'FOO'
            >>> os.environ['FOO'] = 'myvalue'
            >>> s.get_config_variable('foo')
            'myvalue'
            >>> s.set_config_variable('foo', 'othervalue')
            >>> s.get_config_variable('foo')
            'othervalue'

        :type logical_name: str
        :param logical_name: The logical name of the session variable
            you want to set.  These are the keys in ``SESSION_VARIABLES``.

        :param value: The value to associate with the config variable.
        """
        self._overrides[logical_name] = value

    def clear_config_variable(self, logical_name):
        """Remove an override config variable from the session.

        :type logical_name: str
        :param logical_name: The name of the parameter to clear the override
            value from.
        """
        self._overrides.pop(logical_name, None)

    def set_config_provider(self, logical_name, provider):
        """Set the provider for a config value.

        This provides control over how a particular configuration value is
        loaded. This replaces the provider for ``logical_name`` with the new
        ``provider``.

        :type logical_name: str
        :param logical_name: The name of the config value to change the config
            provider for.

        :type provider: :class:`botocore.configprovider.BaseProvider`
        :param provider: The new provider that should be responsible for
            providing a value for the config named ``logical_name``.
        """
        self._mapping[logical_name] = provider