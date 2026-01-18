from pprint import pformat
from six import iteritems
import re
@volume_binding_mode.setter
def volume_binding_mode(self, volume_binding_mode):
    """
        Sets the volume_binding_mode of this V1beta1StorageClass.
        VolumeBindingMode indicates how PersistentVolumeClaims should be
        provisioned and bound.  When unset, VolumeBindingImmediate is used. This
        field is only honored by servers that enable the VolumeScheduling
        feature.

        :param volume_binding_mode: The volume_binding_mode of this
        V1beta1StorageClass.
        :type: str
        """
    self._volume_binding_mode = volume_binding_mode