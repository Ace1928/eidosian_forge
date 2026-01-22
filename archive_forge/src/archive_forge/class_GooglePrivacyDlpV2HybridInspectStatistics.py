from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2HybridInspectStatistics(_messages.Message):
    """Statistics related to processing hybrid inspect requests.

  Fields:
    abortedCount: The number of hybrid inspection requests aborted because the
      job ran out of quota or was ended before they could be processed.
    pendingCount: The number of hybrid requests currently being processed.
      Only populated when called via method `getDlpJob`. A burst of traffic
      may cause hybrid inspect requests to be enqueued. Processing will take
      place as quickly as possible, but resource limitations may impact how
      long a request is enqueued for.
    processedCount: The number of hybrid inspection requests processed within
      this job.
  """
    abortedCount = _messages.IntegerField(1)
    pendingCount = _messages.IntegerField(2)
    processedCount = _messages.IntegerField(3)