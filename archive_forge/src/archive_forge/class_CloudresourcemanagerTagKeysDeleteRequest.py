from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerTagKeysDeleteRequest(_messages.Message):
    """A CloudresourcemanagerTagKeysDeleteRequest object.

  Fields:
    etag: Optional. The etag known to the client for the expected state of the
      TagKey. This is to be used for optimistic concurrency.
    name: Required. The resource name of a TagKey to be deleted in the format
      `tagKeys/123`. The TagKey cannot be a parent of any existing TagValues
      or it will not be deleted successfully.
    validateOnly: Optional. Set as true to perform validations necessary for
      deletion, but not actually perform the action.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)