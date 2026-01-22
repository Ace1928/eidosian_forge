import weakref
from oslo_concurrency import lockutils
from neutron_lib.plugins import constants
class _PluginDirectory(object):
    """A directory of activated plugins in a Neutron Deployment.

    The directory is bootstrapped by a Neutron Manager running in
    the context of a Neutron Server process.
    """

    def __init__(self):
        self._plugins = {}

    def add_plugin(self, alias, plugin):
        """Add a plugin of type 'alias'."""
        self._plugins[alias] = plugin

    def get_plugin(self, alias):
        """Get a plugin for a given alias or None if not present."""
        p = self._plugins.get(alias)
        return weakref.proxy(p) if p else None

    @property
    def plugins(self):
        """The mapping alias -> weak reference to the plugin."""
        return dict(((x, weakref.proxy(y)) for x, y in self._plugins.items()))

    @property
    def unique_plugins(self):
        """A sequence of the unique plugins activated in the environments."""
        return tuple((weakref.proxy(x) for x in set(self._plugins.values())))

    @property
    def is_loaded(self):
        """True if the directory is non empty."""
        return len(self._plugins) > 0