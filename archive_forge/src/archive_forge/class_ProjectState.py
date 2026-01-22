from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProjectState(_messages.Message):
    """Describes a Project State. This contains migration state as well as
  network type of the project

  Enums:
    MigrationStateValueValuesEnum: Output only. The migration state of the
      alias.
    NetworkTypeValueValuesEnum: Output only. The network type of the alias.

  Fields:
    migrationState: Output only. The migration state of the alias.
    name: Output only. Name of the resource which represents the Project state
      of a given project
    networkType: Output only. The network type of the alias.
  """

    class MigrationStateValueValuesEnum(_messages.Enum):
        """Output only. The migration state of the alias.

    Values:
      MIGRATION_STATE_UNSPECIFIED: Default migration state. This value should
        never be used.
      PENDING: Indicates migration is yet to be performed.
      COMPLETED: Indicates migration is successfully completed.
      IN_PROGRESS: Indicates migration in progress.
      NOT_REQUIRED: This value indicates there was no migration state for the
        alias defined in the system. This would be the case for the aliases
        that are new i.e., do not have resource at the time of cutover.
    """
        MIGRATION_STATE_UNSPECIFIED = 0
        PENDING = 1
        COMPLETED = 2
        IN_PROGRESS = 3
        NOT_REQUIRED = 4

    class NetworkTypeValueValuesEnum(_messages.Enum):
        """Output only. The network type of the alias.

    Values:
      NETWORK_TYPE_UNSPECIFIED: Default network type. This value should never
        be used.
      LEGACY: Indicates project is using legacy resources.
      STANDARD: Indicates project is using standard resources.
    """
        NETWORK_TYPE_UNSPECIFIED = 0
        LEGACY = 1
        STANDARD = 2
    migrationState = _messages.EnumField('MigrationStateValueValuesEnum', 1)
    name = _messages.StringField(2)
    networkType = _messages.EnumField('NetworkTypeValueValuesEnum', 3)