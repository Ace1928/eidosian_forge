from pprint import pformat
from six import iteritems
import re
@volume_namespace.setter
def volume_namespace(self, volume_namespace):
    """
        Sets the volume_namespace of this V1StorageOSVolumeSource.
        VolumeNamespace specifies the scope of the volume within StorageOS.  If
        no namespace is specified then the Pod's namespace will be used.  This
        allows the Kubernetes name scoping to be mirrored within StorageOS for
        tighter integration. Set VolumeName to any name to override the default
        behaviour. Set to "default" if you are not using namespaces within
        StorageOS. Namespaces that do not pre-exist within StorageOS will be
        created.

        :param volume_namespace: The volume_namespace of this
        V1StorageOSVolumeSource.
        :type: str
        """
    self._volume_namespace = volume_namespace