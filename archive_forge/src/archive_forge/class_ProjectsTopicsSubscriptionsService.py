from __future__ import absolute_import
import os
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.pubsub_apitools.pubsub_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
from the subscription.
class ProjectsTopicsSubscriptionsService(base_api.BaseApiService):
    """Service class for the projects_topics_subscriptions resource."""
    _NAME = u'projects_topics_subscriptions'

    def __init__(self, client):
        super(PubsubV1.ProjectsTopicsSubscriptionsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the name of the subscriptions for this topic.

      Args:
        request: (PubsubProjectsTopicsSubscriptionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTopicSubscriptionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/topics/{topicsId}/subscriptions', http_method=u'GET', method_id=u'pubsub.projects.topics.subscriptions.list', ordered_params=[u'topic'], path_params=[u'topic'], query_params=[u'pageSize', u'pageToken'], relative_path=u'v1/{+topic}/subscriptions', request_field='', request_type_name=u'PubsubProjectsTopicsSubscriptionsListRequest', response_type_name=u'ListTopicSubscriptionsResponse', supports_download=False)