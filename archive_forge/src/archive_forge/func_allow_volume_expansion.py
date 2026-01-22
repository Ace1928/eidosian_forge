from pprint import pformat
from six import iteritems
import re
@allow_volume_expansion.setter
def allow_volume_expansion(self, allow_volume_expansion):
    """
        Sets the allow_volume_expansion of this V1beta1StorageClass.
        AllowVolumeExpansion shows whether the storage class allow volume expand

        :param allow_volume_expansion: The allow_volume_expansion of this
        V1beta1StorageClass.
        :type: bool
        """
    self._allow_volume_expansion = allow_volume_expansion