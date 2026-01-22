from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsReasoningEnginesGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsReasoningEnginesGetRequest object.

  Fields:
    name: Required. The name of the ReasoningEngine resource. Format: `project
      s/{project}/locations/{location}/reasoningEngines/{reasoning_engine}`
  """
    name = _messages.StringField(1, required=True)