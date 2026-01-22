from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlConnectGetRequest(_messages.Message):
    """A SqlConnectGetRequest object.

  Fields:
    instance: Cloud SQL instance ID. This does not include the project ID.
    project: Project ID of the project that contains the instance.
    readTime: Optional. Optional snapshot read timestamp to trade freshness
      for performance.
  """
    instance = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    readTime = _messages.StringField(3)