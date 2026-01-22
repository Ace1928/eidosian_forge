from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.services import enable_api
import six
def ImagesListView(self):
    """Returns a dictionary representing package vulnerability metadata.

    The returned dictionary is used by artifacts docker images list command.
    """
    messages = ca_requests.GetMessages()
    view = {}
    if self.vulnerabilities:
        view['PACKAGE_VULNERABILITY'] = self.vulnerabilities
    vuln_counts = {}
    for count in self.counts:
        sev = count.severity
        if sev and sev != messages.FixableTotalByDigest.SeverityValueValuesEnum.SEVERITY_UNSPECIFIED:
            vuln_counts.update({sev: vuln_counts.get(sev, 0) + count.totalCount})
    if vuln_counts:
        view['vuln_counts'] = vuln_counts
    return view