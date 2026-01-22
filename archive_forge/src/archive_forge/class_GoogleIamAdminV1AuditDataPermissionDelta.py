from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamAdminV1AuditDataPermissionDelta(_messages.Message):
    """A PermissionDelta message to record the added_permissions and
  removed_permissions inside a role.

  Fields:
    addedPermissions: Added permissions.
    removedPermissions: Removed permissions.
  """
    addedPermissions = _messages.StringField(1, repeated=True)
    removedPermissions = _messages.StringField(2, repeated=True)