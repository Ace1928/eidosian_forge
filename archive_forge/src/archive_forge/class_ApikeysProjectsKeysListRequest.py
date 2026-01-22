from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApikeysProjectsKeysListRequest(_messages.Message):
    """A ApikeysProjectsKeysListRequest object.

  Fields:
    filter: Optional. Only list keys that conform to the given filter. The
      allowed filter strings are `state:ACTIVE` and `state:DELETED`. By
      default, ListKeys will return active keys.
    pageSize: Optional. Specifies the maximum number of results to be returned
      at a time.
    pageToken: Optional. Requests a specific page of results.
    parent: Required. Lists all API keys associated with this project.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)