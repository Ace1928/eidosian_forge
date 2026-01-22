from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsBackupPlansGetRequest(_messages.Message):
    """A GkebackupProjectsLocationsBackupPlansGetRequest object.

  Fields:
    name: Required. Fully qualified BackupPlan name. Format:
      `projects/*/locations/*/backupPlans/*`
  """
    name = _messages.StringField(1, required=True)