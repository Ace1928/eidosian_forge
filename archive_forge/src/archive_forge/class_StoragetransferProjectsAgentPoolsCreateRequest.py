from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StoragetransferProjectsAgentPoolsCreateRequest(_messages.Message):
    """A StoragetransferProjectsAgentPoolsCreateRequest object.

  Fields:
    agentPool: A AgentPool resource to be passed as the request body.
    agentPoolId: Required. The ID of the agent pool to create. The
      `agent_pool_id` must meet the following requirements: * Length of 128
      characters or less. * Not start with the string `goog`. * Start with a
      lowercase ASCII character, followed by: * Zero or more: lowercase Latin
      alphabet characters, numerals, hyphens (`-`), periods (`.`), underscores
      (`_`), or tildes (`~`). * One or more numerals or lowercase ASCII
      characters. As expressed by the regular expression:
      `^(?!goog)[a-z]([a-z0-9-._~]*[a-z0-9])?$`.
    projectId: Required. The ID of the Google Cloud project that owns the
      agent pool.
  """
    agentPool = _messages.MessageField('AgentPool', 1)
    agentPoolId = _messages.StringField(2)
    projectId = _messages.StringField(3, required=True)