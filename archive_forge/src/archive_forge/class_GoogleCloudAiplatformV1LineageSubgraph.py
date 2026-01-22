from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1LineageSubgraph(_messages.Message):
    """A subgraph of the overall lineage graph. Event edges connect Artifact
  and Execution nodes.

  Fields:
    artifacts: The Artifact nodes in the subgraph.
    events: The Event edges between Artifacts and Executions in the subgraph.
    executions: The Execution nodes in the subgraph.
  """
    artifacts = _messages.MessageField('GoogleCloudAiplatformV1Artifact', 1, repeated=True)
    events = _messages.MessageField('GoogleCloudAiplatformV1Event', 2, repeated=True)
    executions = _messages.MessageField('GoogleCloudAiplatformV1Execution', 3, repeated=True)