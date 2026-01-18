import abc
from neutron_lib import constants
@abc.abstractmethod
def unplug(self, device_name, bridge=None, namespace=None, prefix=None):
    """Unplug the interface."""