from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.resource_map import base
from googlecloudsdk.command_lib.util.resource_map.resource_map import ResourceMap
def add_api_to_map(self, api_name, api_versions):
    """Adds an API and all contained resources to the ResourceMap.

    Args:
      api_name: Name of the api to be added.
      api_versions: All registered versions of the api.
    """
    api_data = base.ApiData(api_name, {})
    collection_to_apis_dict = self.get_collection_to_apis_dict(api_name, api_versions)
    for collection_name, supported_apis in collection_to_apis_dict.items():
        api_data.add_resource(base.ResourceData(collection_name, api_name, {'supported_apis': supported_apis}))
    self._resource_map.add_api(api_data)