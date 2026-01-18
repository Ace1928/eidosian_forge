from pprint import pformat
from six import iteritems
import re
@volume_devices.setter
def volume_devices(self, volume_devices):
    """
        Sets the volume_devices of this V1Container.
        volumeDevices is the list of block devices to be used by the container.
        This is a beta feature.

        :param volume_devices: The volume_devices of this V1Container.
        :type: list[V1VolumeDevice]
        """
    self._volume_devices = volume_devices