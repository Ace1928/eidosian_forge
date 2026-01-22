from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelsEvaluationsSlicesListRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelsEvaluationsSlicesListRequest object.

  Fields:
    filter: The standard list filter. * `slice.dimension` - for =.
    pageSize: The standard list page size.
    pageToken: The standard list page token. Typically obtained via
      ListModelEvaluationSlicesResponse.next_page_token of the previous
      ModelService.ListModelEvaluationSlices call.
    parent: Required. The resource name of the ModelEvaluation to list the
      ModelEvaluationSlices from. Format: `projects/{project}/locations/{locat
      ion}/models/{model}/evaluations/{evaluation}`
    readMask: Mask specifying which fields to read.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    readMask = _messages.StringField(5)