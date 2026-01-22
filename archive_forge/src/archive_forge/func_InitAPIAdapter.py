from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import resources as cloud_resources
import six
def InitAPIAdapter(api_version, adapter):
    """Initialize an api adapter.

  Args:
    api_version: the api version we want.
    adapter: the api adapter constructor.
  Returns:
    APIAdapter object.
  """
    api_client = core_apis.GetClientInstance(API_NAME, api_version)
    api_client.check_response_func = api_adapter.CheckResponse
    messages = api_client.MESSAGES_MODULE
    registry = cloud_resources.REGISTRY.Clone()
    registry.RegisterApiByName(API_NAME, api_version)
    return adapter(registry, api_client, messages, api_version)