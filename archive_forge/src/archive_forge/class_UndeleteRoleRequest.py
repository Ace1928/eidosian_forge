from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UndeleteRoleRequest(_messages.Message):
    """The request to undelete an existing role.

  Fields:
    etag: Used to perform a consistent read-modify-write.
  """
    etag = _messages.BytesField(1)