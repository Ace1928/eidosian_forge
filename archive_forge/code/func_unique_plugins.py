import weakref
from oslo_concurrency import lockutils
from neutron_lib.plugins import constants
@property
def unique_plugins(self):
    """A sequence of the unique plugins activated in the environments."""
    return tuple((weakref.proxy(x) for x in set(self._plugins.values())))