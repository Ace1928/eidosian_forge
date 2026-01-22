from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsUpdateLdapssettingsRequest(_messages.Message):
    """A
  ManagedidentitiesProjectsLocationsGlobalDomainsUpdateLdapssettingsRequest
  object.

  Fields:
    lDAPSSettings: A LDAPSSettings resource to be passed as the request body.
    name: The resource name of the LDAPS settings. Uses the form:
      `projects/{project}/locations/{location}/domains/{domain}`.
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field. For the `FieldMask` definition, see
      https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask
  """
    lDAPSSettings = _messages.MessageField('LDAPSSettings', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)