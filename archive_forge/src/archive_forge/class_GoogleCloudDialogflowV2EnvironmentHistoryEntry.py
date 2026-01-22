from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2EnvironmentHistoryEntry(_messages.Message):
    """Represents an environment history entry.

  Fields:
    agentVersion: The agent version loaded into this environment history
      entry.
    createTime: The creation time of this environment history entry.
    description: The developer-provided description for this environment
      history entry.
  """
    agentVersion = _messages.StringField(1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)