from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SharedFlow(_messages.Message):
    """The metadata describing a shared flow

  Fields:
    latestRevisionId: The id of the most recently created revision for this
      shared flow.
    metaData: Metadata describing the shared flow.
    name: The ID of the shared flow.
    revision: A list of revisions of this shared flow.
  """
    latestRevisionId = _messages.StringField(1)
    metaData = _messages.MessageField('GoogleCloudApigeeV1EntityMetadata', 2)
    name = _messages.StringField(3)
    revision = _messages.StringField(4, repeated=True)