from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
class ProjectsLocationsDatasetsHl7V2StoresMessagesService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_hl7V2Stores_messages resource."""
    _NAME = 'projects_locations_datasets_hl7V2Stores_messages'

    def __init__(self, client):
        super(HealthcareV1alpha2.ProjectsLocationsDatasetsHl7V2StoresMessagesService, self).__init__(client)
        self._upload_configs = {}

    def Export(self, request, global_params=None):
        """Exports the messages to a destination in the store with transformations. The start and the end time relative to message generation time (MSH.7) can be specified to filter messages in a range instead of exporting all at once. This API returns an Operation that can be used to track the status of the job by calling GetOperation. Immediate fatal errors appear in the error field. Otherwise, when the operation finishes, a detailed response of type ExportMessagesResponse is returned in the response field. The metadata field type for this operation is OperationMetadata.

      Args:
        request: (HealthcareProjectsLocationsDatasetsHl7V2StoresMessagesExportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Export')
        return self._RunMethod(config, request, global_params=global_params)
    Export.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/hl7V2Stores/{hl7V2StoresId}/messages:export', http_method='POST', method_id='healthcare.projects.locations.datasets.hl7V2Stores.messages.export', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}/messages:export', request_field='exportMessagesRequest', request_type_name='HealthcareProjectsLocationsDatasetsHl7V2StoresMessagesExportRequest', response_type_name='Operation', supports_download=False)