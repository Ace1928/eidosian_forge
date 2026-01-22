from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.services import enable_api
import six
class ContainerAnalysisMetadata:
    """ContainerAnalysisMetadata defines metadata retrieved from containeranalysis API.
  """

    def __init__(self):
        self.vulnerability = PackageVulnerabilitySummary()
        self.image = ImageBasisSummary()
        self.discovery = DiscoverySummary()
        self.deployment = DeploymentSummary()
        self.build = BuildSummary()
        self.provenance = ProvenanceSummary()
        self.package = PackageSummary()
        self.attestation = AttestationSummary()
        self.upgrade = UpgradeSummary()
        self.compliance = ComplianceSummary()
        self.dsse_attestation = DsseAttestaionSummary()
        self.sbom_reference = SbomReferenceSummary()

    def AddOccurrence(self, occ, include_build=True):
        """Adds occurrences retrieved from containeranalysis API.

    Generally we have a 1-1 correspondence between type and summary it's added
    to. The exceptions (due to backwards compatibility issues) are:
    BUILD: If you pass in --show-provenance, there will be a provenance
    section (for intoto builds) but no build section. If you pass in
    --show-all-metadata or --show-build-details, there will be a provenance
    section (for intoto builds) and a builds section (for every build). That
    does mean an occurrence may be in both provenance_summary and build_summary.
    DSSE_ATTESTATION: We always return it in both the DSSE_ATTESTATION section
    and the provenance section.

    Args:
      occ: the occurrence retrieved from the API.
      include_build: whether build-kind occurrences should be added to build.
    """
        messages = ca_requests.GetMessages()
        if occ.kind == messages.Occurrence.KindValueValuesEnum.VULNERABILITY:
            self.vulnerability.AddOccurrence(occ)
        elif occ.kind == messages.Occurrence.KindValueValuesEnum.IMAGE:
            self.image.AddOccurrence(occ)
        elif occ.kind == messages.Occurrence.KindValueValuesEnum.DEPLOYMENT:
            self.deployment.AddOccurrence(occ)
        elif occ.kind == messages.Occurrence.KindValueValuesEnum.DISCOVERY:
            self.discovery.AddOccurrence(occ)
        elif occ.kind == messages.Occurrence.KindValueValuesEnum.DSSE_ATTESTATION:
            self.provenance.AddOccurrence(occ)
        elif occ.kind == messages.Occurrence.KindValueValuesEnum.BUILD and occ.build and (occ.build.intotoStatement or occ.build.inTotoSlsaProvenanceV1):
            self.provenance.AddOccurrence(occ)
        elif occ.kind == messages.Occurrence.KindValueValuesEnum.PACKAGE:
            self.package.AddOccurrence(occ)
        elif occ.kind == messages.Occurrence.KindValueValuesEnum.ATTESTATION:
            self.attestation.AddOccurrence(occ)
        elif occ.kind == messages.Occurrence.KindValueValuesEnum.UPGRADE:
            self.upgrade.AddOccurrence(occ)
        elif occ.kind == messages.Occurrence.KindValueValuesEnum.COMPLIANCE:
            self.compliance.AddOccurrence(occ)
        elif occ.kind == messages.Occurrence.KindValueValuesEnum.SBOM_REFERENCE:
            self.sbom_reference.AddOccurrence(occ)
        if occ.kind == messages.Occurrence.KindValueValuesEnum.DSSE_ATTESTATION:
            self.dsse_attestation.AddOccurrence(occ)
        if occ.kind == messages.Occurrence.KindValueValuesEnum.BUILD and include_build:
            self.build.AddOccurrence(occ)

    def ImagesListView(self):
        """Returns a dictionary representing the metadata.

    The returned dictionary is used by artifacts docker images list command.
    """
        view = {}
        if self.image.base_images:
            view['IMAGE'] = self.image.base_images
        if self.deployment.deployments:
            view['DEPLOYMENT'] = self.deployment.deployments
        if self.discovery.discovery:
            view['DISCOVERY'] = self.discovery.discovery
        if self.build.build_details:
            view['BUILD'] = self.build.build_details
        if self.package.packages:
            view['PACKAGE'] = self.package.packages
        if self.attestation.attestations:
            view['ATTESTATION'] = self.attestation.attestations
        if self.upgrade.upgrades:
            view['UPGRADE'] = self.upgrade.upgrades
        if self.compliance.compliances:
            view['COMPLIANCE'] = self.compliance.compliances
        if self.dsse_attestation.dsse_attestations:
            view['DSSE_ATTESTATION'] = self.dsse_attestation.dsse_attestations
        if self.sbom_reference.sbom_references:
            view['SBOM_REFERENCE'] = self.sbom_reference.sbom_references
        view.update(self.vulnerability.ImagesListView())
        return view

    def ArtifactsDescribeView(self):
        """Returns a dictionary representing the metadata.

    The returned dictionary is used by artifacts docker images describe command.
    """
        view = {}
        if self.image.base_images:
            view['image_basis_summary'] = self.image
        if self.deployment.deployments:
            view['deployment_summary'] = self.deployment
        if self.discovery.discovery:
            view['discovery_summary'] = self.discovery
        if self.build.build_details:
            view['build_details_summary'] = self.build
        vuln = self.vulnerability.ArtifactsDescribeView()
        if vuln:
            view['package_vulnerability_summary'] = vuln
        if self.provenance.provenance:
            view['provenance_summary'] = self.provenance
        if self.package.packages:
            view['package_summary'] = self.package
        if self.attestation.attestations:
            view['attestation_summary'] = self.attestation
        if self.upgrade.upgrades:
            view['upgrade_summary'] = self.upgrade
        if self.compliance.compliances:
            view['compliance_summary'] = self.compliance
        if self.dsse_attestation.dsse_attestations:
            view['dsse_attestation_summary'] = self.dsse_attestation
        if self.sbom_reference.sbom_references:
            view['sbom_summary'] = self.sbom_reference
        return view

    def SLSABuildLevel(self):
        """Returns SLSA build level 0-3 or unknown."""
        if self.provenance.provenance:
            return _ComputeSLSABuildLevel(self.provenance.provenance)
        return 'unknown'

    def SbomLocations(self):
        return [sbom_ref.sbomReference.payload.predicate.location for sbom_ref in self.sbom_reference.sbom_references]