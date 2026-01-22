from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocParameters(_messages.Message):
    """Parameters used in Dataproc JobType executions.

  Fields:
    cluster: URI for cluster used to run Dataproc execution. Format:
      `projects/{PROJECT_ID}/regions/{REGION}/clusters/{CLUSTER_NAME}`
  """
    cluster = _messages.StringField(1)