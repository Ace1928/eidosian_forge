from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerIODetails(_messages.Message):
    """Metadata for a Spanner connector used by the job.

  Fields:
    databaseId: DatabaseId accessed in the connection.
    instanceId: InstanceId accessed in the connection.
    projectId: ProjectId accessed in the connection.
  """
    databaseId = _messages.StringField(1)
    instanceId = _messages.StringField(2)
    projectId = _messages.StringField(3)