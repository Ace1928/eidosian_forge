from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentEnvironmentsUsersSessionsDetectIntentRequest(_messages.Message):
    """A DialogflowProjectsAgentEnvironmentsUsersSessionsDetectIntentRequest
  object.

  Fields:
    googleCloudDialogflowV2DetectIntentRequest: A
      GoogleCloudDialogflowV2DetectIntentRequest resource to be passed as the
      request body.
    session: Required. The name of the session this query is sent to. Format:
      `projects//agent/sessions/`, or
      `projects//agent/environments//users//sessions/`. If `Environment ID` is
      not specified, we assume default 'draft' environment (`Environment ID`
      might be referred to as environment name at some places). If `User ID`
      is not specified, we are using "-". It's up to the API caller to choose
      an appropriate `Session ID` and `User Id`. They can be a random number
      or some type of user and session identifiers (preferably hashed). The
      length of the `Session ID` and `User ID` must not exceed 36 characters.
      For more information, see the [API interactions
      guide](https://cloud.google.com/dialogflow/docs/api-overview). Note:
      Always use agent versions for production traffic. See [Versions and
      environments](https://cloud.google.com/dialogflow/es/docs/agents-
      versions).
  """
    googleCloudDialogflowV2DetectIntentRequest = _messages.MessageField('GoogleCloudDialogflowV2DetectIntentRequest', 1)
    session = _messages.StringField(2, required=True)