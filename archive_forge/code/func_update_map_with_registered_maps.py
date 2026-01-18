from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.resource_map import base
from googlecloudsdk.command_lib.util.resource_map.resource_map import ResourceMap
def update_map_with_registered_maps(self):
    """Updates resource map using registered resource maps.

    This will iterate through each registered resource map and apply any
    contained metadata to the resource map. All registered resource maps must
    have an analogous structure to the underlying resource map.
    """
    for update_map in self._update_maps:
        for api in self._resource_map:
            api_name = api.get_api_name()
            for resource in api:
                resource_name = resource.get_resource_name()
                if api_name in update_map and resource_name in update_map[api_name]:
                    for key, value in update_map[api_name][resource_name].items():
                        setattr(resource, key, value)