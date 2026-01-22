from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentGetValidationResultRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentGetValidationResultRequest object.

  Fields:
    languageCode: Optional. The language for which you want a validation
      result. If not specified, the agent's default language is used. [Many
      languages](https://cloud.google.com/dialogflow/docs/reference/language)
      are supported. Note: languages must be enabled in the agent before they
      can be used.
    parent: Required. The project that the agent is associated with. Format:
      `projects/`.
  """
    languageCode = _messages.StringField(1)
    parent = _messages.StringField(2, required=True)