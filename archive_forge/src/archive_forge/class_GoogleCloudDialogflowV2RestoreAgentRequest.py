from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2RestoreAgentRequest(_messages.Message):
    """The request message for Agents.RestoreAgent.

  Fields:
    agentContent: Zip compressed raw byte content for agent.
    agentUri: The URI to a Google Cloud Storage file containing the agent to
      restore. Note: The URI must start with "gs://". Dialogflow performs a
      read operation for the Cloud Storage object on the caller's behalf, so
      your request authentication must have read permissions for the object.
      For more information, see [Dialogflow access
      control](https://cloud.google.com/dialogflow/cx/docs/concept/access-
      control#storage).
  """
    agentContent = _messages.BytesField(1)
    agentUri = _messages.StringField(2)