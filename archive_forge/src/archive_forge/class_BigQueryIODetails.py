from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigQueryIODetails(_messages.Message):
    """Metadata for a BigQuery connector used by the job.

  Fields:
    dataset: Dataset accessed in the connection.
    projectId: Project accessed in the connection.
    query: Query used to access data in the connection.
    table: Table accessed in the connection.
  """
    dataset = _messages.StringField(1)
    projectId = _messages.StringField(2)
    query = _messages.StringField(3)
    table = _messages.StringField(4)