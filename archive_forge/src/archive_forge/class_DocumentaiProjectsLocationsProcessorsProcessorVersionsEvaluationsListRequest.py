from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorsProcessorVersionsEvaluationsListRequest(_messages.Message):
    """A
  DocumentaiProjectsLocationsProcessorsProcessorVersionsEvaluationsListRequest
  object.

  Fields:
    pageSize: The standard list page size. If unspecified, at most `5`
      evaluations are returned. The maximum value is `100`. Values above `100`
      are coerced to `100`.
    pageToken: A page token, received from a previous `ListEvaluations` call.
      Provide this to retrieve the subsequent page.
    parent: Required. The resource name of the ProcessorVersion to list
      evaluations for. `projects/{project}/locations/{location}/processors/{pr
      ocessor}/processorVersions/{processorVersion}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)