from pprint import pformat
from six import iteritems
import re
@allowed_csi_drivers.setter
def allowed_csi_drivers(self, allowed_csi_drivers):
    """
        Sets the allowed_csi_drivers of this PolicyV1beta1PodSecurityPolicySpec.
        AllowedCSIDrivers is a whitelist of inline CSI drivers that must be
        explicitly set to be embedded within a pod spec. An empty value means no
        CSI drivers can run inline within a pod spec.

        :param allowed_csi_drivers: The allowed_csi_drivers of this
        PolicyV1beta1PodSecurityPolicySpec.
        :type: list[PolicyV1beta1AllowedCSIDriver]
        """
    self._allowed_csi_drivers = allowed_csi_drivers