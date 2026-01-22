from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsub.v1 import pubsub_v1_messages as messages
class ProjectsTopicsSnapshotsService(base_api.BaseApiService):
    """Service class for the projects_topics_snapshots resource."""
    _NAME = 'projects_topics_snapshots'

    def __init__(self, client):
        super(PubsubV1.ProjectsTopicsSnapshotsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the names of the snapshots on this topic. Snapshots are used in [Seek](https://cloud.google.com/pubsub/docs/replay-overview) operations, which allow you to manage message acknowledgments in bulk. That is, you can set the acknowledgment state of messages in an existing subscription to the state captured by a snapshot.

      Args:
        request: (PubsubProjectsTopicsSnapshotsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTopicSnapshotsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/topics/{topicsId}/snapshots', http_method='GET', method_id='pubsub.projects.topics.snapshots.list', ordered_params=['topic'], path_params=['topic'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+topic}/snapshots', request_field='', request_type_name='PubsubProjectsTopicsSnapshotsListRequest', response_type_name='ListTopicSnapshotsResponse', supports_download=False)