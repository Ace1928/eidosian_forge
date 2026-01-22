from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1beta1 import healthcare_v1beta1_messages as messages
def GetStorageInfo(self, request, global_params=None):
    """GetStorageInfo returns the storage info of the specified resource.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesSeriesInstancesGetStorageInfoRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StorageInfo) The response message.
      """
    config = self.GetMethodConfig('GetStorageInfo')
    return self._RunMethod(config, request, global_params=global_params)