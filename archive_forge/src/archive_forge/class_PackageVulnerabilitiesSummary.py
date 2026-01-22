from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.container.images import container_data_util
from googlecloudsdk.api_lib.containeranalysis import requests
import six
class PackageVulnerabilitiesSummary(SummaryResolver):
    """PackageVulnerabilitiesSummary has information about vulnerabilities."""

    def __init__(self):
        self.__messages = requests.GetMessages()
        self.vulnerabilities = collections.defaultdict(list)

    def add_record(self, occ):
        sev = six.text_type(occ.vulnerability.effectiveSeverity)
        self.vulnerabilities[sev].append(occ)

    def resolve(self):
        self.total_vulnerability_found = 0
        self.not_fixed_vulnerability_count = 0
        for occs in self.vulnerabilities.values():
            for occ in occs:
                self.total_vulnerability_found += 1
                for package_issue in occ.vulnerability.packageIssue:
                    if package_issue.fixedVersion.kind == self.__messages.Version.KindValueValuesEnum.MAXIMUM:
                        self.not_fixed_vulnerability_count += 1
                        break
        self.vulnerabilities = dict(self.vulnerabilities)