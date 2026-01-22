from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PackageOccurrence(_messages.Message):
    """Details on how a particular software package was installed on a system.

  Enums:
    ArchitectureValueValuesEnum: Output only. The CPU architecture for which
      packages in this distribution channel were built. Architecture will be
      blank for language packages.

  Fields:
    architecture: Output only. The CPU architecture for which packages in this
      distribution channel were built. Architecture will be blank for language
      packages.
    cpeUri: Output only. The cpe_uri in [CPE
      format](https://cpe.mitre.org/specification/) denoting the package
      manager version distributing a package. The cpe_uri will be blank for
      language packages.
    license: Licenses that have been declared by the authors of the package.
    location: All of the places within the filesystem versions of this package
      have been found.
    name: Required. Output only. The name of the installed package.
    packageType: Output only. The type of package; whether native or non
      native (e.g., ruby gems, node.js packages, etc.).
    version: Output only. The version of the package.
  """

    class ArchitectureValueValuesEnum(_messages.Enum):
        """Output only. The CPU architecture for which packages in this
    distribution channel were built. Architecture will be blank for language
    packages.

    Values:
      ARCHITECTURE_UNSPECIFIED: Unknown architecture.
      X86: X86 architecture.
      X64: X64 architecture.
    """
        ARCHITECTURE_UNSPECIFIED = 0
        X86 = 1
        X64 = 2
    architecture = _messages.EnumField('ArchitectureValueValuesEnum', 1)
    cpeUri = _messages.StringField(2)
    license = _messages.MessageField('License', 3)
    location = _messages.MessageField('Location', 4, repeated=True)
    name = _messages.StringField(5)
    packageType = _messages.StringField(6)
    version = _messages.MessageField('Version', 7)