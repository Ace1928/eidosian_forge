from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1AsyncQueryResult(_messages.Message):
    """A GoogleCloudApigeeV1AsyncQueryResult object.

  Fields:
    expires: Query result will be unaccessable after this time.
    self: Self link of the query results. Example: `/organizations/myorg/envir
      onments/myenv/queries/9cfc0d85-0f30-46d6-ae6f-318d0cb961bd/result` or
      following format if query is running at host level: `/organizations/myor
      g/hostQueries/9cfc0d85-0f30-46d6-ae6f-318d0cb961bd/result`
  """
    expires = _messages.StringField(1)
    self = _messages.StringField(2)