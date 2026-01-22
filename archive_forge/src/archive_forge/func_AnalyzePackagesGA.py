from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def AnalyzePackagesGA(project, location, resource_uri, packages):
    """Make an RPC to the On-Demand Scanning v1 AnalyzePackages method."""
    client = GetClient('v1')
    messages = GetMessages('v1')
    req = messages.OndemandscanningProjectsLocationsScansAnalyzePackagesRequest(parent=PARENT_TEMPLATE.format(project, location), analyzePackagesRequestV1=messages.AnalyzePackagesRequestV1(packages=packages, resourceUri=resource_uri))
    return client.projects_locations_scans.AnalyzePackages(req)