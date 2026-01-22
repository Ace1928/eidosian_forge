from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpecificDomainControllers(_messages.Message):
    """Configuration of specific domain controllers.

  Fields:
    primaryServerUri: Required. Primary domain controller LDAP server for the
      domain. Format `ldap://hostname:port` or `ldaps://hostname:port`.
    secondaryServerUri: Optional. Secondary domain controller LDAP server for
      the domain. Format `ldap://hostname:port` or `ldaps://hostname:port`.
  """
    primaryServerUri = _messages.StringField(1)
    secondaryServerUri = _messages.StringField(2)