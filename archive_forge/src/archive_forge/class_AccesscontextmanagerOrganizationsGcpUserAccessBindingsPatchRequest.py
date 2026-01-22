from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerOrganizationsGcpUserAccessBindingsPatchRequest(_messages.Message):
    """A AccesscontextmanagerOrganizationsGcpUserAccessBindingsPatchRequest
  object.

  Fields:
    gcpUserAccessBinding: A GcpUserAccessBinding resource to be passed as the
      request body.
    name: Immutable. Assigned by the server during creation. The last segment
      has an arbitrary length and has only URI unreserved characters (as
      defined by [RFC 3986 Section
      2.3](https://tools.ietf.org/html/rfc3986#section-2.3)). Should not be
      specified by the client during creation. Example:
      "organizations/256/gcpUserAccessBindings/b3-BhcX_Ud5N"
    updateMask: Required. Only the fields specified in this mask are updated.
      Because name and group_key cannot be changed, update_mask is required
      and may only contain the following fields: `access_levels`,
      `dry_run_access_levels`, `restricted_client_applications`,
      `reauth_settings`. Example: update_mask { paths: "access_levels" }
  """
    gcpUserAccessBinding = _messages.MessageField('GcpUserAccessBinding', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)