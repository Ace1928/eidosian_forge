from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerAutoHealingPolicyAutoHealingTriggers(_messages.Message):
    """A InstanceGroupManagerAutoHealingPolicyAutoHealingTriggers object.

  Enums:
    OnHealthCheckValueValuesEnum: If you have configured an application-based
      health check for the group, this field controls whether to trigger VM
      autohealing based on a failed health check. Valid values are: - ON
      (default): The group recreates running VMs that fail the application-
      based health check. - OFF: When set to OFF, you can still observe
      instance health state, but the group does not recreate VMs that fail the
      application-based health check. This is useful for troubleshooting and
      setting up your health check configuration.

  Fields:
    onHealthCheck: If you have configured an application-based health check
      for the group, this field controls whether to trigger VM autohealing
      based on a failed health check. Valid values are: - ON (default): The
      group recreates running VMs that fail the application-based health
      check. - OFF: When set to OFF, you can still observe instance health
      state, but the group does not recreate VMs that fail the application-
      based health check. This is useful for troubleshooting and setting up
      your health check configuration.
  """

    class OnHealthCheckValueValuesEnum(_messages.Enum):
        """If you have configured an application-based health check for the
    group, this field controls whether to trigger VM autohealing based on a
    failed health check. Valid values are: - ON (default): The group recreates
    running VMs that fail the application-based health check. - OFF: When set
    to OFF, you can still observe instance health state, but the group does
    not recreate VMs that fail the application-based health check. This is
    useful for troubleshooting and setting up your health check configuration.

    Values:
      OFF: When set to OFF, you can still observe instance health state, but
        the group does not recreate VMs that fail the application-based health
        check. This is useful for troubleshooting and setting up your health
        check configuration.
      ON: (Default) The group recreates running VMs that fail the group's
        application-based health check.
    """
        OFF = 0
        ON = 1
    onHealthCheck = _messages.EnumField('OnHealthCheckValueValuesEnum', 1)