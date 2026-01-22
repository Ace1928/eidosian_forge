from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from apitools.base.py import  exceptions as apitools_exc
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.generated_clients.apis import apis_map
import six
def GetMethods(full_collection_name, api_version=None, disable_pagination=False):
    """Gets all the methods available on the given collection.

  Args:
    full_collection_name: str, The collection including the api name.
    api_version: str, The version string of the API or None to use the default
      for this API.
    disable_pagination: bool, Boolean for whether pagination should be disabled

  Returns:
    [APIMethod], The method specifications.
  """
    api_collection = GetAPICollection(full_collection_name, api_version=api_version)
    client = _GetApiClient(api_collection.api_name, api_collection.api_version)
    service = _GetService(client, api_collection.name)
    if not service:
        return []
    method_names = service.GetMethodsList()
    method_configs = [(name, service.GetMethodConfig(name)) for name in method_names]
    return [APIMethod(service, name, api_collection, config, disable_pagination) for name, config in method_configs]