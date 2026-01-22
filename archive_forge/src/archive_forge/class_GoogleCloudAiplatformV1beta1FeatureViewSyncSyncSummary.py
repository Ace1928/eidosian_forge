from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeatureViewSyncSyncSummary(_messages.Message):
    """Summary from the Sync job. For continuous syncs, the summary is updated
  periodically. For batch syncs, it gets updated on completion of the sync.

  Fields:
    rowSynced: Output only. Total number of rows synced.
    totalSlot: Output only. BigQuery slot milliseconds consumed for the sync
      job.
  """
    rowSynced = _messages.IntegerField(1)
    totalSlot = _messages.IntegerField(2)