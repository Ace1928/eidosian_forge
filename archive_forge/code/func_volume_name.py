from pprint import pformat
from six import iteritems
import re
@volume_name.setter
def volume_name(self, volume_name):
    """
        Sets the volume_name of this V1StorageOSVolumeSource.
        VolumeName is the human-readable name of the StorageOS volume.  Volume
        names are only unique within a namespace.

        :param volume_name: The volume_name of this V1StorageOSVolumeSource.
        :type: str
        """
    self._volume_name = volume_name