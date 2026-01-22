from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastreamProjectsLocationsStreamsObjectsStopBackfillJobRequest(_messages.Message):
    """A DatastreamProjectsLocationsStreamsObjectsStopBackfillJobRequest
  object.

  Fields:
    object: Required. The name of the stream object resource to stop the
      backfill job for.
    stopBackfillJobRequest: A StopBackfillJobRequest resource to be passed as
      the request body.
  """
    object = _messages.StringField(1, required=True)
    stopBackfillJobRequest = _messages.MessageField('StopBackfillJobRequest', 2)