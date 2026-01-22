from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsProvidersListRequest(_messages.Message):
    """A ConnectorsProjectsLocationsProvidersListRequest object.

  Fields:
    pageSize: Page size.
    pageToken: Page token.
    parent: Required. Parent resource of the API, of the form:
      `projects/*/locations/*` Only global location is supported for Provider
      resource.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)