from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def traffic_manager_endpoint_to_dict(endpoint):
    return dict(id=endpoint.id, name=endpoint.name, type=endpoint.type, target_resource_id=endpoint.target_resource_id, target=endpoint.target, status=endpoint.endpoint_status, weight=endpoint.weight, priority=endpoint.priority, location=endpoint.endpoint_location, monitor_status=endpoint.endpoint_monitor_status, min_child_endpoints=endpoint.min_child_endpoints, geo_mapping=endpoint.geo_mapping)