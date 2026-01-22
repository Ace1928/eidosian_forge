from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiscoveryNote(_messages.Message):
    """A note that indicates a type of analysis a provider would perform. This
  note exists in a provider's project. A `Discovery` occurrence is created in
  a consumer's project at the start of analysis.

  Enums:
    AnalysisKindValueValuesEnum: Required. Immutable. The kind of analysis
      that is handled by this discovery.

  Fields:
    analysisKind: Required. Immutable. The kind of analysis that is handled by
      this discovery.
  """

    class AnalysisKindValueValuesEnum(_messages.Enum):
        """Required. Immutable. The kind of analysis that is handled by this
    discovery.

    Values:
      NOTE_KIND_UNSPECIFIED: Default value. This value is unused.
      VULNERABILITY: The note and occurrence represent a package
        vulnerability.
      BUILD: The note and occurrence assert build provenance.
      IMAGE: This represents an image basis relationship.
      PACKAGE: This represents a package installed via a package manager.
      DEPLOYMENT: The note and occurrence track deployment events.
      DISCOVERY: The note and occurrence track the initial discovery status of
        a resource.
      ATTESTATION: This represents a logical "role" that can attest to
        artifacts.
      UPGRADE: This represents an available package upgrade.
      COMPLIANCE: This represents a Compliance Note
      DSSE_ATTESTATION: This represents a DSSE attestation Note
      VULNERABILITY_ASSESSMENT: This represents a Vulnerability Assessment.
      SBOM_REFERENCE: This represents an SBOM Reference.
    """
        NOTE_KIND_UNSPECIFIED = 0
        VULNERABILITY = 1
        BUILD = 2
        IMAGE = 3
        PACKAGE = 4
        DEPLOYMENT = 5
        DISCOVERY = 6
        ATTESTATION = 7
        UPGRADE = 8
        COMPLIANCE = 9
        DSSE_ATTESTATION = 10
        VULNERABILITY_ASSESSMENT = 11
        SBOM_REFERENCE = 12
    analysisKind = _messages.EnumField('AnalysisKindValueValuesEnum', 1)