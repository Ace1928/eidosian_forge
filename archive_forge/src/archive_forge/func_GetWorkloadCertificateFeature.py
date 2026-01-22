from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.workloadcertificate.v1alpha1 import workloadcertificate_v1alpha1_messages as messages
def GetWorkloadCertificateFeature(self, request, global_params=None):
    """Gets the `WorkloadCertificateFeature` resource of a given project. `WorkloadCertificateFeature` is a singleton resource.

      Args:
        request: (WorkloadcertificateProjectsLocationsGlobalGetWorkloadCertificateFeatureRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkloadCertificateFeature) The response message.
      """
    config = self.GetMethodConfig('GetWorkloadCertificateFeature')
    return self._RunMethod(config, request, global_params=global_params)