from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsBackupPlansCreateRequest(_messages.Message):
    """A GkebackupProjectsLocationsBackupPlansCreateRequest object.

  Fields:
    backupPlan: A BackupPlan resource to be passed as the request body.
    backupPlanId: Required. The client-provided short name for the BackupPlan
      resource. This name must: - be between 1 and 63 characters long
      (inclusive) - consist of only lower-case ASCII letters, numbers, and
      dashes - start with a lower-case letter - end with a lower-case letter
      or number - be unique within the set of BackupPlans in this location
    parent: Required. The location within which to create the BackupPlan.
      Format: `projects/*/locations/*`
  """
    backupPlan = _messages.MessageField('BackupPlan', 1)
    backupPlanId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)