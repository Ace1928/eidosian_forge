from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.services import enable_api
import six
def GetMavenArtifactOccurrences(project, maven_resource):
    """Retrieves occurrences for Maven artifacts."""
    metadata = ContainerAnalysisMetadata()
    occ_filter = _CreateFilterForMaven(maven_resource)
    occurrences = ca_requests.ListOccurrences(project, occ_filter)
    for occ in occurrences:
        metadata.AddOccurrence(occ, False)
    return metadata