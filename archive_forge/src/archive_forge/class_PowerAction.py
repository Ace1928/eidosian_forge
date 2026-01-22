import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
class PowerAction(enum.Enum):
    """Mapping from an action to a target power state."""
    POWER_ON = 'power on'
    'Power on the node.'
    POWER_OFF = 'power off'
    'Power off the node (using hard power off).'
    REBOOT = 'rebooting'
    'Reboot the node (using hard power off).'
    SOFT_POWER_OFF = 'soft power off'
    'Power off the node using soft power off.'
    SOFT_REBOOT = 'soft rebooting'
    'Reboot the node using soft power off.'