from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.container.images import container_data_util
from googlecloudsdk.api_lib.containeranalysis import requests
import six
class ContainerAndAnalysisData(container_data_util.ContainerData):
    """Class defining container and analysis data.

  ContainerAndAnalysisData subclasses ContainerData because we want it to
  contain a superset of the attributes, particularly when `--format=json`,
  `format=value(digest)`, etc. is used with `container images describe`.
  """

    def __init__(self, name):
        super(ContainerAndAnalysisData, self).__init__(registry=name.registry, repository=name.repository, digest=name.digest)
        self.package_vulnerability_summary = PackageVulnerabilitiesSummary()
        self.image_basis_summary = ImageBasesSummary()
        self.build_details_summary = BuildsSummary()
        self.deployment_summary = DeploymentsSummary()
        self.discovery_summary = DiscoverySummary()

    def add_record(self, occurrence):
        messages = requests.GetMessages()
        if occurrence.kind == messages.Occurrence.KindValueValuesEnum.VULNERABILITY:
            self.package_vulnerability_summary.add_record(occurrence)
        elif occurrence.kind == messages.Occurrence.KindValueValuesEnum.IMAGE:
            self.image_basis_summary.add_record(occurrence)
        elif occurrence.kind == messages.Occurrence.KindValueValuesEnum.BUILD:
            self.build_details_summary.add_record(occurrence)
        elif occurrence.kind == messages.Occurrence.KindValueValuesEnum.DEPLOYMENT:
            self.deployment_summary.add_record(occurrence)
        elif occurrence.kind == messages.Occurrence.KindValueValuesEnum.DISCOVERY:
            self.discovery_summary.add_record(occurrence)

    def resolveSummaries(self):
        self.package_vulnerability_summary.resolve()
        self.image_basis_summary.resolve()
        self.build_details_summary.resolve()
        self.deployment_summary.resolve()
        self.discovery_summary.resolve()