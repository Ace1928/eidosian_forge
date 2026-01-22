from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsResourceValueConfigsPatchRequest(_messages.Message):
    """A SecuritycenterOrganizationsResourceValueConfigsPatchRequest object.

  Fields:
    googleCloudSecuritycenterV2ResourceValueConfig: A
      GoogleCloudSecuritycenterV2ResourceValueConfig resource to be passed as
      the request body.
    name: Name for the resource value config
    updateMask: The list of fields to be updated. If empty all mutable fields
      will be updated.
  """
    googleCloudSecuritycenterV2ResourceValueConfig = _messages.MessageField('GoogleCloudSecuritycenterV2ResourceValueConfig', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)