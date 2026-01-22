from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsReasoningEnginesListRequest(_messages.Message):
    """A AiplatformProjectsLocationsReasoningEnginesListRequest object.

  Fields:
    filter: Optional. The standard list filter. More detail in
      [AIP-160](https://google.aip.dev/160).
    pageSize: Optional. The standard list page size.
    pageToken: Optional. The standard list page token.
    parent: Required. The resource name of the Location to list the
      ReasoningEngines from. Format: `projects/{project}/locations/{location}`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)