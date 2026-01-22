from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorsProcessorVersionsEvaluationsGetRequest(_messages.Message):
    """A
  DocumentaiProjectsLocationsProcessorsProcessorVersionsEvaluationsGetRequest
  object.

  Fields:
    name: Required. The resource name of the Evaluation to get. `projects/{pro
      ject}/locations/{location}/processors/{processor}/processorVersions/{pro
      cessorVersion}/evaluations/{evaluation}`
  """
    name = _messages.StringField(1, required=True)