from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BulkRestoreObjectsRequest(_messages.Message):
    """A bulk restore objects request.

  Fields:
    allowOverwrite: If false (default), the restore will not overwrite live
      objects with the same name at the destination. This means some deleted
      objects may be skipped. If true, live objects will be overwritten
      resulting in a noncurrent object (if versioning is enabled). If
      versioning is not enabled, overwriting the object will result in a soft-
      deleted object. In either case, if a noncurrent object already exists
      with the same name, a live version can be written without issue.
    copySourceAcl: If true, copies the source object's ACL; otherwise, uses
      the bucket's default object ACL. The default is false.
    matchGlobs: Restores only the objects matching any of the specified
      glob(s). If this parameter is not specified, all objects will be
      restored within the specified time range.
    softDeletedAfterTime: Restores only the objects that were soft-deleted
      after this time.
    softDeletedBeforeTime: Restores only the objects that were soft-deleted
      before this time.
  """
    allowOverwrite = _messages.BooleanField(1)
    copySourceAcl = _messages.BooleanField(2)
    matchGlobs = _messages.StringField(3, repeated=True)
    softDeletedAfterTime = _message_types.DateTimeField(4)
    softDeletedBeforeTime = _message_types.DateTimeField(5)