from pprint import pformat
from six import iteritems
import re
@allowed_flex_volumes.setter
def allowed_flex_volumes(self, allowed_flex_volumes):
    """
        Sets the allowed_flex_volumes of this
        PolicyV1beta1PodSecurityPolicySpec.
        allowedFlexVolumes is a whitelist of allowed Flexvolumes.  Empty or nil
        indicates that all Flexvolumes may be used.  This parameter is effective
        only when the usage of the Flexvolumes is allowed in the "volumes"
        field.

        :param allowed_flex_volumes: The allowed_flex_volumes of this
        PolicyV1beta1PodSecurityPolicySpec.
        :type: list[PolicyV1beta1AllowedFlexVolume]
        """
    self._allowed_flex_volumes = allowed_flex_volumes