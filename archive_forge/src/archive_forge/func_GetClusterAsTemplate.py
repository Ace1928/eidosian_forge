from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def GetClusterAsTemplate(self, request, global_params=None):
    """Exports a template for a cluster in a project that can be used in future CreateCluster requests.

      Args:
        request: (DataprocProjectsRegionsClustersGetClusterAsTemplateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Cluster) The response message.
      """
    config = self.GetMethodConfig('GetClusterAsTemplate')
    return self._RunMethod(config, request, global_params=global_params)