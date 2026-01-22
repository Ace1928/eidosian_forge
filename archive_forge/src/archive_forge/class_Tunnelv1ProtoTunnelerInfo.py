from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Tunnelv1ProtoTunnelerInfo(_messages.Message):
    """TunnelerInfo contains metadata about tunneler launched by connection
  manager.

  Fields:
    backoffRetryCount: backoff_retry_count stores the number of times the
      tunneler has been retried by tunManager for current backoff sequence.
      Gets reset to 0 if time difference between 2 consecutive retries exceeds
      backoffRetryResetTime.
    id: id is the unique id of a tunneler.
    latestErr: latest_err stores the Error for the latest tunneler failure.
      Gets reset everytime the tunneler is retried by tunManager.
    latestRetryTime: latest_retry_time stores the time when the tunneler was
      last restarted.
    totalRetryCount: total_retry_count stores the total number of times the
      tunneler has been retried by tunManager.
  """
    backoffRetryCount = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    id = _messages.StringField(2)
    latestErr = _messages.MessageField('Tunnelv1ProtoTunnelerError', 3)
    latestRetryTime = _messages.StringField(4)
    totalRetryCount = _messages.IntegerField(5, variant=_messages.Variant.UINT32)