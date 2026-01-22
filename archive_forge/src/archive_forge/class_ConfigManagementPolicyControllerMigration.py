from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigManagementPolicyControllerMigration(_messages.Message):
    """State for the migration of PolicyController from ACM -> PoCo Hub.

  Enums:
    StageValueValuesEnum: Stage of the migration.

  Fields:
    copyTime: Last time this membership spec was copied to PoCo feature.
    stage: Stage of the migration.
  """

    class StageValueValuesEnum(_messages.Enum):
        """Stage of the migration.

    Values:
      STAGE_UNSPECIFIED: Unknown state of migration.
      ACM_MANAGED: ACM Hub/Operator manages policycontroller. No migration yet
        completed.
      POCO_MANAGED: All migrations steps complete; Poco Hub now manages
        policycontroller.
    """
        STAGE_UNSPECIFIED = 0
        ACM_MANAGED = 1
        POCO_MANAGED = 2
    copyTime = _messages.StringField(1)
    stage = _messages.EnumField('StageValueValuesEnum', 2)