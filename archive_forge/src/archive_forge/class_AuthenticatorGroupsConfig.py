from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthenticatorGroupsConfig(_messages.Message):
    """Configuration for returning group information from authenticators.

  Fields:
    enabled: Whether this cluster should return group membership lookups
      during authentication using a group of security groups.
    securityGroup: The name of the security group-of-groups to be used. Only
      relevant if enabled = true.
  """
    enabled = _messages.BooleanField(1)
    securityGroup = _messages.StringField(2)