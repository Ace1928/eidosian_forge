from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchProjectsLocationsResourceAllowancesListRequest(_messages.Message):
    """A BatchProjectsLocationsResourceAllowancesListRequest object.

  Fields:
    pageSize: Optional. Page size.
    pageToken: Optional. Page token.
    parent: Required. Parent path.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)