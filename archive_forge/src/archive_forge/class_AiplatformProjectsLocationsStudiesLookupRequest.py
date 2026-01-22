from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsStudiesLookupRequest(_messages.Message):
    """A AiplatformProjectsLocationsStudiesLookupRequest object.

  Fields:
    googleCloudAiplatformV1LookupStudyRequest: A
      GoogleCloudAiplatformV1LookupStudyRequest resource to be passed as the
      request body.
    parent: Required. The resource name of the Location to get the Study from.
      Format: `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1LookupStudyRequest = _messages.MessageField('GoogleCloudAiplatformV1LookupStudyRequest', 1)
    parent = _messages.StringField(2, required=True)