from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreateSessionsRequest(_messages.Message):
    """The request for BatchCreateSessions.

  Fields:
    sessionCount: Required. The number of sessions to be created in this batch
      call. The API may return fewer than the requested number of sessions. If
      a specific number of sessions are desired, the client can make
      additional calls to BatchCreateSessions (adjusting session_count as
      necessary).
    sessionTemplate: Parameters to be applied to each created session.
  """
    sessionCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    sessionTemplate = _messages.MessageField('Session', 2)