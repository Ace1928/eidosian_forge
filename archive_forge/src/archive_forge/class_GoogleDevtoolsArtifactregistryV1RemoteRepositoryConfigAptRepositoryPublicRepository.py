from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigAptRepositoryPublicRepository(_messages.Message):
    """Publicly available Apt repositories constructed from a common repository
  base and a custom repository path.

  Enums:
    RepositoryBaseValueValuesEnum: A common public repository base for Apt.

  Fields:
    repositoryBase: A common public repository base for Apt.
    repositoryPath: A custom field to define a path to a specific repository
      from the base.
  """

    class RepositoryBaseValueValuesEnum(_messages.Enum):
        """A common public repository base for Apt.

    Values:
      REPOSITORY_BASE_UNSPECIFIED: Unspecified repository base.
      DEBIAN: Debian.
      UBUNTU: Ubuntu LTS/Pro.
      DEBIAN_SNAPSHOT: Archived Debian.
    """
        REPOSITORY_BASE_UNSPECIFIED = 0
        DEBIAN = 1
        UBUNTU = 2
        DEBIAN_SNAPSHOT = 3
    repositoryBase = _messages.EnumField('RepositoryBaseValueValuesEnum', 1)
    repositoryPath = _messages.StringField(2)