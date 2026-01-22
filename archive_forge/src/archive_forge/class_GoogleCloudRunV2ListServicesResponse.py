from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2ListServicesResponse(_messages.Message):
    """Response message containing a list of Services.

  Fields:
    nextPageToken: A token indicating there are more items than page_size. Use
      it in the next ListServices request to continue.
    services: The resulting list of Services.
  """
    nextPageToken = _messages.StringField(1)
    services = _messages.MessageField('GoogleCloudRunV2Service', 2, repeated=True)