from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class YumArtifact(_messages.Message):
    """A detailed representation of a Yum artifact.

  Enums:
    PackageTypeValueValuesEnum: Output only. An artifact is a binary or source
      package.

  Fields:
    architecture: Output only. Operating system architecture of the artifact.
    name: Output only. The Artifact Registry resource name of the artifact.
    packageName: Output only. The yum package name of the artifact.
    packageType: Output only. An artifact is a binary or source package.
  """

    class PackageTypeValueValuesEnum(_messages.Enum):
        """Output only. An artifact is a binary or source package.

    Values:
      PACKAGE_TYPE_UNSPECIFIED: Package type is not specified.
      BINARY: Binary package (.rpm).
      SOURCE: Source package (.srpm).
    """
        PACKAGE_TYPE_UNSPECIFIED = 0
        BINARY = 1
        SOURCE = 2
    architecture = _messages.StringField(1)
    name = _messages.StringField(2)
    packageName = _messages.StringField(3)
    packageType = _messages.EnumField('PackageTypeValueValuesEnum', 4)