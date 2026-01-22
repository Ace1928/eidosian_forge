from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceStatusServiceIntegrationStatusBackupDRStatus(_messages.Message):
    """Message defining compute perspective of the result of integration with
  Backup and DR. FAILED status indicates that the operation specified did not
  complete correctly and should be retried with the same value.

  Enums:
    StateValueValuesEnum: Enum representing the registration state of a Backup
      and DR backup plan for the instance.

  Fields:
    integrationDetails: The PlanReference object created by Backup and DR to
      maintain the actual status of backups. May still be present if removing
      the backup plan fails.
    state: Enum representing the registration state of a Backup and DR backup
      plan for the instance.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Enum representing the registration state of a Backup and DR backup
    plan for the instance.

    Values:
      ACTIVE: The operation was successful and Backup and DR is trying to
        protect the instance with the specified backup plan. Check resource
        pointed to in integration_details for more information.
      CREATING: GCE is trying to attach the backup plan to the instance.
      DELETING: GCE is trying to remove the backup plan from the instance.
      FAILED: The operation failed, specifying the same value in
        BackupDrSpec.plan again (including null for delete) will attempt to
        repair the integration
      STATE_UNSPECIFIED: Default value, should not be found on instances.
    """
        ACTIVE = 0
        CREATING = 1
        DELETING = 2
        FAILED = 3
        STATE_UNSPECIFIED = 4
    integrationDetails = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)