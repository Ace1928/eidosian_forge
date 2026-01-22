from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerOrganizationsGcpUserAccessBindingsCreateRequest(_messages.Message):
    """A AccesscontextmanagerOrganizationsGcpUserAccessBindingsCreateRequest
  object.

  Fields:
    gcpUserAccessBinding: A GcpUserAccessBinding resource to be passed as the
      request body.
    parent: Required. Example: "organizations/256"
  """
    gcpUserAccessBinding = _messages.MessageField('GcpUserAccessBinding', 1)
    parent = _messages.StringField(2, required=True)