from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsRevokeRequest(_messages.Message):
    """A PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsRevokeRequest
  object.

  Fields:
    name: Required. Name of the Grant resource which is being revoked.
    revokeGrantRequest: A RevokeGrantRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    revokeGrantRequest = _messages.MessageField('RevokeGrantRequest', 2)