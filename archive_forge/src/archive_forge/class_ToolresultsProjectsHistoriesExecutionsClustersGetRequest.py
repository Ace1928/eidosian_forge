from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesExecutionsClustersGetRequest(_messages.Message):
    """A ToolresultsProjectsHistoriesExecutionsClustersGetRequest object.

  Fields:
    clusterId: A Cluster id Required.
    executionId: An Execution id. Required.
    historyId: A History id. Required.
    projectId: A Project id. Required.
  """
    clusterId = _messages.StringField(1, required=True)
    executionId = _messages.StringField(2, required=True)
    historyId = _messages.StringField(3, required=True)
    projectId = _messages.StringField(4, required=True)