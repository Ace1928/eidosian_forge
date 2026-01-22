from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MavenRepositoryConfig(_messages.Message):
    """MavenRepositoryConfig is maven related repository details. Provides
  additional configuration details for repositories of the maven format type.

  Enums:
    VersionPolicyValueValuesEnum: Version policy defines the versions that the
      registry will accept.

  Fields:
    allowSnapshotOverwrites: The repository with this flag will allow
      publishing the same snapshot versions.
    versionPolicy: Version policy defines the versions that the registry will
      accept.
  """

    class VersionPolicyValueValuesEnum(_messages.Enum):
        """Version policy defines the versions that the registry will accept.

    Values:
      VERSION_POLICY_UNSPECIFIED: VERSION_POLICY_UNSPECIFIED - the version
        policy is not defined. When the version policy is not defined, no
        validation is performed for the versions.
      RELEASE: RELEASE - repository will accept only Release versions.
      SNAPSHOT: SNAPSHOT - repository will accept only Snapshot versions.
    """
        VERSION_POLICY_UNSPECIFIED = 0
        RELEASE = 1
        SNAPSHOT = 2
    allowSnapshotOverwrites = _messages.BooleanField(1)
    versionPolicy = _messages.EnumField('VersionPolicyValueValuesEnum', 2)