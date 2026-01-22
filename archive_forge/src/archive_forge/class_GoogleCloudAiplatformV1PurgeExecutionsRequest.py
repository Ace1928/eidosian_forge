from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1PurgeExecutionsRequest(_messages.Message):
    """Request message for MetadataService.PurgeExecutions.

  Fields:
    filter: Required. A required filter matching the Executions to be purged.
      E.g., `update_time <= 2020-11-19T11:30:00-04:00`.
    force: Optional. Flag to indicate to actually perform the purge. If
      `force` is set to false, the method will return a sample of Execution
      names that would be deleted.
  """
    filter = _messages.StringField(1)
    force = _messages.BooleanField(2)