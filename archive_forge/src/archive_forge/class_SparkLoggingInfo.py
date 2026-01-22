from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkLoggingInfo(_messages.Message):
    """Spark job logs can be filtered by these fields in Cloud Logging.

  Fields:
    projectId: Output only. Project ID where the Spark logs were written.
    resourceType: Output only. Resource type used for logging.
  """
    projectId = _messages.StringField(1)
    resourceType = _messages.StringField(2)