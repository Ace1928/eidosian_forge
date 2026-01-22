from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.workflowexecutions.v1 import workflowexecutions_v1_messages as messages
def ExportData(self, request, global_params=None):
    """Returns all metadata stored about an execution, excluding most data that is already accessible using other API methods.

      Args:
        request: (WorkflowexecutionsProjectsLocationsWorkflowsExecutionsExportDataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExportDataResponse) The response message.
      """
    config = self.GetMethodConfig('ExportData')
    return self._RunMethod(config, request, global_params=global_params)