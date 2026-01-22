from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSchemasDeleteRequest(_messages.Message):
    """A PubsubProjectsSchemasDeleteRequest object.

  Fields:
    name: Required. Name of the schema to delete. Format is
      `projects/{project}/schemas/{schema}`.
  """
    name = _messages.StringField(1, required=True)