from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsStudiesTrialsListRequest(_messages.Message):
    """A AiplatformProjectsLocationsStudiesTrialsListRequest object.

  Fields:
    pageSize: Optional. The number of Trials to retrieve per "page" of
      results. If unspecified, the service will pick an appropriate default.
    pageToken: Optional. A page token to request the next page of results. If
      unspecified, there are no subsequent pages.
    parent: Required. The resource name of the Study to list the Trial from.
      Format: `projects/{project}/locations/{location}/studies/{study}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)