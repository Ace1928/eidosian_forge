from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsReasoningEnginesDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsReasoningEnginesDeleteRequest object.

  Fields:
    name: Required. The name of the ReasoningEngine resource to be deleted.
      Format: `projects/{project}/locations/{location}/reasoningEngines/{reaso
      ning_engine}`
  """
    name = _messages.StringField(1, required=True)