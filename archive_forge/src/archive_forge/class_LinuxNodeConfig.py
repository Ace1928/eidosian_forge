from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinuxNodeConfig(_messages.Message):
    """Parameters that can be configured on Linux nodes.

  Enums:
    CgroupModeValueValuesEnum: cgroup_mode specifies the cgroup mode to be
      used on the node.

  Messages:
    SysctlsValue: The Linux kernel parameters to be applied to the nodes and
      all pods running on the nodes. The following parameters are supported.
      net.core.busy_poll net.core.busy_read net.core.netdev_max_backlog
      net.core.rmem_max net.core.wmem_default net.core.wmem_max
      net.core.optmem_max net.core.somaxconn net.ipv4.tcp_rmem
      net.ipv4.tcp_wmem net.ipv4.tcp_tw_reuse

  Fields:
    cgroupMode: cgroup_mode specifies the cgroup mode to be used on the node.
    hugepages: Optional. Amounts for 2M and 1G hugepages
    sysctls: The Linux kernel parameters to be applied to the nodes and all
      pods running on the nodes. The following parameters are supported.
      net.core.busy_poll net.core.busy_read net.core.netdev_max_backlog
      net.core.rmem_max net.core.wmem_default net.core.wmem_max
      net.core.optmem_max net.core.somaxconn net.ipv4.tcp_rmem
      net.ipv4.tcp_wmem net.ipv4.tcp_tw_reuse
  """

    class CgroupModeValueValuesEnum(_messages.Enum):
        """cgroup_mode specifies the cgroup mode to be used on the node.

    Values:
      CGROUP_MODE_UNSPECIFIED: CGROUP_MODE_UNSPECIFIED is when unspecified
        cgroup configuration is used. The default for the GKE node OS image
        will be used.
      CGROUP_MODE_V1: CGROUP_MODE_V1 specifies to use cgroupv1 for the cgroup
        configuration on the node image.
      CGROUP_MODE_V2: CGROUP_MODE_V2 specifies to use cgroupv2 for the cgroup
        configuration on the node image.
    """
        CGROUP_MODE_UNSPECIFIED = 0
        CGROUP_MODE_V1 = 1
        CGROUP_MODE_V2 = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SysctlsValue(_messages.Message):
        """The Linux kernel parameters to be applied to the nodes and all pods
    running on the nodes. The following parameters are supported.
    net.core.busy_poll net.core.busy_read net.core.netdev_max_backlog
    net.core.rmem_max net.core.wmem_default net.core.wmem_max
    net.core.optmem_max net.core.somaxconn net.ipv4.tcp_rmem net.ipv4.tcp_wmem
    net.ipv4.tcp_tw_reuse

    Messages:
      AdditionalProperty: An additional property for a SysctlsValue object.

    Fields:
      additionalProperties: Additional properties of type SysctlsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SysctlsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    cgroupMode = _messages.EnumField('CgroupModeValueValuesEnum', 1)
    hugepages = _messages.MessageField('HugepagesConfig', 2)
    sysctls = _messages.MessageField('SysctlsValue', 3)