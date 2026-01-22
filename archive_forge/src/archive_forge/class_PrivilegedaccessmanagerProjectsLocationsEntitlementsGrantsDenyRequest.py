from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivilegedaccessmanagerProjectsLocationsEntitlementsGrantsDenyRequest(_messages.Message):
    """A PrivilegedaccessmanagerProjectsLocationsEntitlementsGrantsDenyRequest
  object.

  Fields:
    denyGrantRequest: A DenyGrantRequest resource to be passed as the request
      body.
    name: Required. Name of the Grant resource which is being denied.
  """
    denyGrantRequest = _messages.MessageField('DenyGrantRequest', 1)
    name = _messages.StringField(2, required=True)