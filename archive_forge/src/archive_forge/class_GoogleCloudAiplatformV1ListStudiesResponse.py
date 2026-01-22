from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListStudiesResponse(_messages.Message):
    """Response message for VizierService.ListStudies.

  Fields:
    nextPageToken: Passes this token as the `page_token` field of the request
      for a subsequent call. If this field is omitted, there are no subsequent
      pages.
    studies: The studies associated with the project.
  """
    nextPageToken = _messages.StringField(1)
    studies = _messages.MessageField('GoogleCloudAiplatformV1Study', 2, repeated=True)