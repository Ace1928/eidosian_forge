from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadConfig(_messages.Message):
    """WorkloadConfig defines the flags to enable or disable the workload
  configurations for the cluster.

  Enums:
    AuditModeValueValuesEnum: Sets which mode of auditing should be used for
      the cluster's workloads.
    VulnerabilityScanningModeValueValuesEnum: Sets which mode of vulnerability
      scanning should be used for cluster's workloads.

  Fields:
    auditMode: Sets which mode of auditing should be used for the cluster's
      workloads.
    vulnerabilityScanningMode: Sets which mode of vulnerability scanning
      should be used for cluster's workloads.
  """

    class AuditModeValueValuesEnum(_messages.Enum):
        """Sets which mode of auditing should be used for the cluster's
    workloads.

    Values:
      MODE_UNSPECIFIED: Default value meaning that no mode has been specified.
      DISABLED: This disables Workload Configuration auditing on the cluster,
        meaning that nothing is surfaced.
      BASIC: Applies the default set of policy auditing to a cluster's
        workloads.
      BASELINE: Surfaces configurations that are not in line with the Pod
        Security Standard Baseline policy.
      RESTRICTED: Surfaces configurations that are not in line with the Pod
        Security Standard Restricted policy.
    """
        MODE_UNSPECIFIED = 0
        DISABLED = 1
        BASIC = 2
        BASELINE = 3
        RESTRICTED = 4

    class VulnerabilityScanningModeValueValuesEnum(_messages.Enum):
        """Sets which mode of vulnerability scanning should be used for cluster's
    workloads.

    Values:
      MODE_UNSPECIFIED: Default value meaning that no mode has been specified.
      DISABLED: This disables Workload Configuration auditing on the cluster,
        meaning that nothing is surfaced.
      BASIC: Applies the default set of policy auditing to a cluster's
        workloads.
      BASELINE: Surfaces configurations that are not in line with the Pod
        Security Standard Baseline policy.
      RESTRICTED: Surfaces configurations that are not in line with the Pod
        Security Standard Restricted policy.
    """
        MODE_UNSPECIFIED = 0
        DISABLED = 1
        BASIC = 2
        BASELINE = 3
        RESTRICTED = 4
    auditMode = _messages.EnumField('AuditModeValueValuesEnum', 1)
    vulnerabilityScanningMode = _messages.EnumField('VulnerabilityScanningModeValueValuesEnum', 2)