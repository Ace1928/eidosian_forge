from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AptRepository(_messages.Message):
    """Represents a single Apt package repository. This repository is added to
  a repo file that is stored at
  `/etc/apt/sources.list.d/google_osconfig.list`.

  Enums:
    ArchiveTypeValueValuesEnum: Type of archive files in this repository. The
      default behavior is DEB.

  Fields:
    archiveType: Type of archive files in this repository. The default
      behavior is DEB.
    components: Required. List of components for this repository. Must contain
      at least one item.
    distribution: Required. Distribution of this repository.
    gpgKey: URI of the key file for this repository. The agent maintains a
      keyring at `/etc/apt/trusted.gpg.d/osconfig_agent_managed.gpg`
      containing all the keys in any applied guest policy.
    uri: Required. URI for this repository.
  """

    class ArchiveTypeValueValuesEnum(_messages.Enum):
        """Type of archive files in this repository. The default behavior is DEB.

    Values:
      ARCHIVE_TYPE_UNSPECIFIED: Unspecified.
      DEB: DEB indicates that the archive contains binary files.
      DEB_SRC: DEB_SRC indicates that the archive contains source files.
    """
        ARCHIVE_TYPE_UNSPECIFIED = 0
        DEB = 1
        DEB_SRC = 2
    archiveType = _messages.EnumField('ArchiveTypeValueValuesEnum', 1)
    components = _messages.StringField(2, repeated=True)
    distribution = _messages.StringField(3)
    gpgKey = _messages.StringField(4)
    uri = _messages.StringField(5)