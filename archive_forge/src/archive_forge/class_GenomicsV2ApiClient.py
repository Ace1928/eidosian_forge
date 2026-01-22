from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
from googlecloudsdk.api_lib.genomics import genomics_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
import six
class GenomicsV2ApiClient(GenomicsApiClient):
    """Client for accessing the V2 genomics API.
  """

    def __init__(self):
        super(GenomicsV2ApiClient, self).__init__('v2alpha1')

    def ResourceFromName(self, name):
        return self._registry.Parse(name, collection='genomics.projects.operations', params={'projectsId': genomics_util.GetProjectId()})

    def Poller(self):
        return waiter.CloudOperationPollerNoResources(self._client.projects_operations)

    def GetOperation(self, resource):
        return self._client.projects_operations.Get(self._messages.GenomicsProjectsOperationsGetRequest(name=resource.RelativeName()))

    def CancelOperation(self, resource):
        return self._client.projects_operations.Cancel(self._messages.GenomicsProjectsOperationsCancelRequest(name=resource.RelativeName()))