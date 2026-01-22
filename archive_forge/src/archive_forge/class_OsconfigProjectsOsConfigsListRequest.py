from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsOsConfigsListRequest(_messages.Message):
    """A OsconfigProjectsOsConfigsListRequest object.

  Fields:
    pageSize: The maximum number of OsConfigs to return.
    pageToken: A pagination token returned from a previous call to
      ListOsConfigs that indicates where this listing should continue from.
      This field is optional.
    parent: The resource name of the parent.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)