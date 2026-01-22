from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApikeysProjectsLocationsKeysListRequest(_messages.Message):
    """A ApikeysProjectsLocationsKeysListRequest object.

  Fields:
    pageSize: Optional. Specifies the maximum number of results to be returned
      at a time.
    pageToken: Optional. Requests a specific page of results.
    parent: Required. Lists all API keys associated with this project.
    showDeleted: Optional. Indicate that keys deleted in the past 30 days
      should also be returned.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    showDeleted = _messages.BooleanField(4)