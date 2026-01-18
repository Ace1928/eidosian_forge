from pprint import pformat
from six import iteritems
import re
@volume_id.setter
def volume_id(self, volume_id):
    """
        Sets the volume_id of this V1PortworxVolumeSource.
        VolumeID uniquely identifies a Portworx volume

        :param volume_id: The volume_id of this V1PortworxVolumeSource.
        :type: str
        """
    if volume_id is None:
        raise ValueError('Invalid value for `volume_id`, must not be `None`')
    self._volume_id = volume_id