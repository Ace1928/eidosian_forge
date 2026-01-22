from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcpIamAccess(_messages.Message):
    """GcpIamAccess represents IAM based access control on a GCP resource.
  Refer to https://cloud.google.com/iam/docs to understand more about IAM.

  Fields:
    resource: Required. Name of the resource.
    resourceType: Required. The type of this resource.
    roleBindings: Required. Role bindings to be created on successful grant.
  """
    resource = _messages.StringField(1)
    resourceType = _messages.StringField(2)
    roleBindings = _messages.MessageField('RoleBinding', 3, repeated=True)