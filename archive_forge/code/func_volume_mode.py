from pprint import pformat
from six import iteritems
import re
@volume_mode.setter
def volume_mode(self, volume_mode):
    """
        Sets the volume_mode of this V1PersistentVolumeSpec.
        volumeMode defines if a volume is intended to be used with a formatted
        filesystem or to remain in raw block state. Value of Filesystem is
        implied when not included in spec. This is a beta feature.

        :param volume_mode: The volume_mode of this V1PersistentVolumeSpec.
        :type: str
        """
    self._volume_mode = volume_mode