from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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