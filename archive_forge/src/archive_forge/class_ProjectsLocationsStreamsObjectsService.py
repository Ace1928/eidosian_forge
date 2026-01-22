from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datastream.v1 import datastream_v1_messages as messages
class ProjectsLocationsStreamsObjectsService(base_api.BaseApiService):
    """Service class for the projects_locations_streams_objects resource."""
    _NAME = 'projects_locations_streams_objects'

    def __init__(self, client):
        super(DatastreamV1.ProjectsLocationsStreamsObjectsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Use this method to get details about a stream object.

      Args:
        request: (DatastreamProjectsLocationsStreamsObjectsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StreamObject) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streams/{streamsId}/objects/{objectsId}', http_method='GET', method_id='datastream.projects.locations.streams.objects.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DatastreamProjectsLocationsStreamsObjectsGetRequest', response_type_name='StreamObject', supports_download=False)

    def List(self, request, global_params=None):
        """Use this method to list the objects of a specific stream.

      Args:
        request: (DatastreamProjectsLocationsStreamsObjectsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListStreamObjectsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streams/{streamsId}/objects', http_method='GET', method_id='datastream.projects.locations.streams.objects.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/objects', request_field='', request_type_name='DatastreamProjectsLocationsStreamsObjectsListRequest', response_type_name='ListStreamObjectsResponse', supports_download=False)

    def Lookup(self, request, global_params=None):
        """Use this method to look up a stream object by its source object identifier.

      Args:
        request: (DatastreamProjectsLocationsStreamsObjectsLookupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StreamObject) The response message.
      """
        config = self.GetMethodConfig('Lookup')
        return self._RunMethod(config, request, global_params=global_params)
    Lookup.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streams/{streamsId}/objects:lookup', http_method='POST', method_id='datastream.projects.locations.streams.objects.lookup', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/objects:lookup', request_field='lookupStreamObjectRequest', request_type_name='DatastreamProjectsLocationsStreamsObjectsLookupRequest', response_type_name='StreamObject', supports_download=False)

    def StartBackfillJob(self, request, global_params=None):
        """Use this method to start a backfill job for the specified stream object.

      Args:
        request: (DatastreamProjectsLocationsStreamsObjectsStartBackfillJobRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StartBackfillJobResponse) The response message.
      """
        config = self.GetMethodConfig('StartBackfillJob')
        return self._RunMethod(config, request, global_params=global_params)
    StartBackfillJob.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streams/{streamsId}/objects/{objectsId}:startBackfillJob', http_method='POST', method_id='datastream.projects.locations.streams.objects.startBackfillJob', ordered_params=['object'], path_params=['object'], query_params=[], relative_path='v1/{+object}:startBackfillJob', request_field='startBackfillJobRequest', request_type_name='DatastreamProjectsLocationsStreamsObjectsStartBackfillJobRequest', response_type_name='StartBackfillJobResponse', supports_download=False)

    def StopBackfillJob(self, request, global_params=None):
        """Use this method to stop a backfill job for the specified stream object.

      Args:
        request: (DatastreamProjectsLocationsStreamsObjectsStopBackfillJobRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StopBackfillJobResponse) The response message.
      """
        config = self.GetMethodConfig('StopBackfillJob')
        return self._RunMethod(config, request, global_params=global_params)
    StopBackfillJob.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streams/{streamsId}/objects/{objectsId}:stopBackfillJob', http_method='POST', method_id='datastream.projects.locations.streams.objects.stopBackfillJob', ordered_params=['object'], path_params=['object'], query_params=[], relative_path='v1/{+object}:stopBackfillJob', request_field='stopBackfillJobRequest', request_type_name='DatastreamProjectsLocationsStreamsObjectsStopBackfillJobRequest', response_type_name='StopBackfillJobResponse', supports_download=False)