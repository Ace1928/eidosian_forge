from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstancesSetSecurityPolicyRequest(_messages.Message):
    """A InstancesSetSecurityPolicyRequest object.

  Fields:
    networkInterfaces: The network interfaces that the security policy will be
      applied to. Network interfaces use the nicN naming format. You can only
      set a security policy for network interfaces with an access config.
    securityPolicy: A full or partial URL to a security policy to add to this
      instance. If this field is set to an empty string it will remove the
      associated security policy.
  """
    networkInterfaces = _messages.StringField(1, repeated=True)
    securityPolicy = _messages.StringField(2)