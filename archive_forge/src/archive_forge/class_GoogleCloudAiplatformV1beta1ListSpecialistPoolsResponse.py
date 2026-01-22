from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListSpecialistPoolsResponse(_messages.Message):
    """Response message for SpecialistPoolService.ListSpecialistPools.

  Fields:
    nextPageToken: The standard List next-page token.
    specialistPools: A list of SpecialistPools that matches the specified
      filter in the request.
  """
    nextPageToken = _messages.StringField(1)
    specialistPools = _messages.MessageField('GoogleCloudAiplatformV1beta1SpecialistPool', 2, repeated=True)