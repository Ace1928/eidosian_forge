from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesBackupSchedulesGetRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesBackupSchedulesGetRequest object.

  Fields:
    name: Required. The name of the schedule to retrieve. Values are of the
      form `projects//instances//databases//backupSchedules/`.
  """
    name = _messages.StringField(1, required=True)