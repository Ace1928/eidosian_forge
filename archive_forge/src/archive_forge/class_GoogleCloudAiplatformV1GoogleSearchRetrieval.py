from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1GoogleSearchRetrieval(_messages.Message):
    """Tool to retrieve public web data for grounding, powered by Google.

  Fields:
    disableAttribution: Optional. Disable using the result from this tool in
      detecting grounding attribution. This does not affect how the result is
      given to the model for generation.
  """
    disableAttribution = _messages.BooleanField(1)