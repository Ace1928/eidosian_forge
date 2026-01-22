from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivilegedaccessmanagerFoldersLocationsEntitlementsPatchRequest(_messages.Message):
    """A PrivilegedaccessmanagerFoldersLocationsEntitlementsPatchRequest
  object.

  Fields:
    entitlement: A Entitlement resource to be passed as the request body.
    name: Identifier. Name of the Entitlement. Possible formats: *
      `organizations/{organization-
      number}/locations/{region}/entitlements/{entitlement-id}` *
      `folders/{folder-number}/locations/{region}/entitlements/{entitlement-
      id}` * `projects/{project-id|project-
      number}/locations/{region}/entitlements/{entitlement-id}`
    updateMask: Required. The list of fields to update. A field will be
      overwritten if, and only if, it is in the mask. Any immutable fields set
      in the mask will be ignored by the server. Repeated fields and map
      fields are only allowed in the last position of a `paths` string and
      will overwrite the existing values. Hence an update to a repeated field
      or a map should contain the entire list of values. The fields specified
      in the update_mask are relative to the resource and not to the request.
      (e.g. `MaxRequestDuration`; *not* `entitlement.MaxRequestDuration`) A
      value of '*' for this field refers to full replacement of the resource.
  """
    entitlement = _messages.MessageField('Entitlement', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)