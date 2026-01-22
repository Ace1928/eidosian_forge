from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceIntegrationSpecBackupDRSpec(_messages.Message):
    """Specifies parameters to Backup and DR to attach a BackupPlan to a
  compute instance for managed VM backup.

  Fields:
    plan: The BackupPlan resource to attach to the instance. Specified as a
      resource reference in instances, and regional instance templates, and as
      just the plan name in global instance templates
  """
    plan = _messages.StringField(1)