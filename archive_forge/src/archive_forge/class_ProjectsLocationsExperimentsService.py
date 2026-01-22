from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.faultinjectiontesting.v1alpha1 import faultinjectiontesting_v1alpha1_messages as messages
class ProjectsLocationsExperimentsService(base_api.BaseApiService):
    """Service class for the projects_locations_experiments resource."""
    _NAME = 'projects_locations_experiments'

    def __init__(self, client):
        super(FaultinjectiontestingV1alpha1.ProjectsLocationsExperimentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Experiment in a given project and location.

      Args:
        request: (FaultinjectiontestingProjectsLocationsExperimentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Experiment) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/experiments', http_method='POST', method_id='faultinjectiontesting.projects.locations.experiments.create', ordered_params=['parent'], path_params=['parent'], query_params=['experimentId', 'requestId'], relative_path='v1alpha1/{+parent}/experiments', request_field='experiment', request_type_name='FaultinjectiontestingProjectsLocationsExperimentsCreateRequest', response_type_name='Experiment', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Experiment.

      Args:
        request: (FaultinjectiontestingProjectsLocationsExperimentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/experiments/{experimentsId}', http_method='DELETE', method_id='faultinjectiontesting.projects.locations.experiments.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='FaultinjectiontestingProjectsLocationsExperimentsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Experiment.

      Args:
        request: (FaultinjectiontestingProjectsLocationsExperimentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Experiment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/experiments/{experimentsId}', http_method='GET', method_id='faultinjectiontesting.projects.locations.experiments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='FaultinjectiontestingProjectsLocationsExperimentsGetRequest', response_type_name='Experiment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Experiments in a given project and location.

      Args:
        request: (FaultinjectiontestingProjectsLocationsExperimentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListExperimentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/experiments', http_method='GET', method_id='faultinjectiontesting.projects.locations.experiments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/experiments', request_field='', request_type_name='FaultinjectiontestingProjectsLocationsExperimentsListRequest', response_type_name='ListExperimentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Experiment.

      Args:
        request: (FaultinjectiontestingProjectsLocationsExperimentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Experiment) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/experiments/{experimentsId}', http_method='PATCH', method_id='faultinjectiontesting.projects.locations.experiments.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha1/{+name}', request_field='experiment', request_type_name='FaultinjectiontestingProjectsLocationsExperimentsPatchRequest', response_type_name='Experiment', supports_download=False)