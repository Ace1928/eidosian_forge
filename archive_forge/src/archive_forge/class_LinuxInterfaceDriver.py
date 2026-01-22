import abc
from neutron_lib import constants
class LinuxInterfaceDriver(object, metaclass=abc.ABCMeta):
    DEV_NAME_LEN = constants.LINUX_DEV_LEN
    DEV_NAME_PREFIX = constants.TAP_DEVICE_PREFIX

    @abc.abstractmethod
    def plug_new(self, network_id, port_id, device_name, mac_address, bridge=None, namespace=None, prefix=None, mtu=None, link_up=True):
        """Plug in the interface only for new devices that don't exist yet."""

    @abc.abstractmethod
    def unplug(self, device_name, bridge=None, namespace=None, prefix=None):
        """Unplug the interface."""