from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import service as recommender_service
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
def GetDescribeConfigRequestFromArgs(parent_resource, is_insight_api, api_version):
    """Returns the describe request from the user-specified arguments.

  Args:
    parent_resource: resource url string, the flags are already defined in
      argparse namespace.
    is_insight_api: boolean value specifying whether this is a insight api,
      otherwise treat as a recommender service api and return related describe
      request message.
    api_version: API version string.
  """
    messages = recommender_service.RecommenderMessages(api_version)
    if is_insight_api:
        request = messages.RecommenderProjectsLocationsInsightTypesGetConfigRequest(name=parent_resource)
    else:
        request = messages.RecommenderProjectsLocationsRecommendersGetConfigRequest(name=parent_resource)
    return request