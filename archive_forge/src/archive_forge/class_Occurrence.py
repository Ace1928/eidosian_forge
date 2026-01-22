from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Occurrence(_messages.Message):
    """An instance of an analysis type that has been found on a resource.

  Enums:
    KindValueValuesEnum: Output only. This explicitly denotes which of the
      occurrence details are specified. This field can be used as a filter in
      list requests.

  Fields:
    attestation: Describes an attestation of an artifact.
    build: Describes a verifiable build.
    compliance: Describes a compliance violation on a linked resource.
    createTime: Output only. The time this occurrence was created.
    deployment: Describes the deployment of an artifact on a runtime.
    discovery: Describes when a resource was discovered.
    dsseAttestation: Describes an attestation of an artifact using dsse.
    envelope: https://github.com/secure-systems-lab/dsse
    image: Describes how this resource derives from the basis in the
      associated note.
    kind: Output only. This explicitly denotes which of the occurrence details
      are specified. This field can be used as a filter in list requests.
    name: Output only. The name of the occurrence in the form of
      `projects/[PROJECT_ID]/occurrences/[OCCURRENCE_ID]`.
    noteName: Required. Immutable. The analysis note associated with this
      occurrence, in the form of `projects/[PROVIDER_ID]/notes/[NOTE_ID]`.
      This field can be used as a filter in list requests.
    package: Describes the installation of a package on the linked resource.
    remediation: A description of actions that can be taken to remedy the
      note.
    resourceUri: Required. Immutable. A URI that represents the resource for
      which the occurrence applies. For example,
      `https://gcr.io/project/image@sha256:123abc` for a Docker image.
    sbomReference: Describes a specific SBOM reference occurrences.
    updateTime: Output only. The time this occurrence was last updated.
    upgrade: Describes an available package upgrade on the linked resource.
    vulnerability: Describes a security vulnerability.
  """

    class KindValueValuesEnum(_messages.Enum):
        """Output only. This explicitly denotes which of the occurrence details
    are specified. This field can be used as a filter in list requests.

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
    attestation = _messages.MessageField('AttestationOccurrence', 1)
    build = _messages.MessageField('BuildOccurrence', 2)
    compliance = _messages.MessageField('ComplianceOccurrence', 3)
    createTime = _messages.StringField(4)
    deployment = _messages.MessageField('DeploymentOccurrence', 5)
    discovery = _messages.MessageField('DiscoveryOccurrence', 6)
    dsseAttestation = _messages.MessageField('DSSEAttestationOccurrence', 7)
    envelope = _messages.MessageField('Envelope', 8)
    image = _messages.MessageField('ImageOccurrence', 9)
    kind = _messages.EnumField('KindValueValuesEnum', 10)
    name = _messages.StringField(11)
    noteName = _messages.StringField(12)
    package = _messages.MessageField('PackageOccurrence', 13)
    remediation = _messages.StringField(14)
    resourceUri = _messages.StringField(15)
    sbomReference = _messages.MessageField('SBOMReferenceOccurrence', 16)
    updateTime = _messages.StringField(17)
    upgrade = _messages.MessageField('UpgradeOccurrence', 18)
    vulnerability = _messages.MessageField('VulnerabilityOccurrence', 19)