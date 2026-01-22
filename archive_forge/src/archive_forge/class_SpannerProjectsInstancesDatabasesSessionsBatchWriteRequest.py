from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesSessionsBatchWriteRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesSessionsBatchWriteRequest object.

  Fields:
    batchWriteRequest: A BatchWriteRequest resource to be passed as the
      request body.
    session: Required. The session in which the batch request is to be run.
  """
    batchWriteRequest = _messages.MessageField('BatchWriteRequest', 1)
    session = _messages.StringField(2, required=True)