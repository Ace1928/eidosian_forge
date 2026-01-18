from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def traffic_manager_profile_to_dict(tmp):
    result = dict(id=tmp.id, name=tmp.name, type=tmp.type, tags=tmp.tags, location=tmp.location, profile_status=tmp.profile_status, traffic_routing_method=tmp.traffic_routing_method, dns_config=dict(), monitor_config=dict(), endpoints=[])
    if tmp.dns_config:
        result['dns_config']['relative_name'] = tmp.dns_config.relative_name
        result['dns_config']['fqdn'] = tmp.dns_config.fqdn
        result['dns_config']['ttl'] = tmp.dns_config.ttl
    if tmp.monitor_config:
        result['monitor_config']['profile_monitor_status'] = tmp.monitor_config.profile_monitor_status
        result['monitor_config']['protocol'] = tmp.monitor_config.protocol
        result['monitor_config']['port'] = tmp.monitor_config.port
        result['monitor_config']['path'] = tmp.monitor_config.path
        result['monitor_config']['interval_in_seconds'] = tmp.monitor_config.interval_in_seconds
        result['monitor_config']['timeout_in_seconds'] = tmp.monitor_config.timeout_in_seconds
        result['monitor_config']['tolerated_number_of_failures'] = tmp.monitor_config.tolerated_number_of_failures
    if tmp.endpoints:
        for endpoint in tmp.endpoints:
            result['endpoints'].append(dict(id=endpoint.id, name=endpoint.name, type=endpoint.type, target_resource_id=endpoint.target_resource_id, target=endpoint.target, endpoint_status=endpoint.endpoint_status, weight=endpoint.weight, priority=endpoint.priority, endpoint_location=endpoint.endpoint_location, endpoint_monitor_status=endpoint.endpoint_monitor_status, min_child_endpoints=endpoint.min_child_endpoints, geo_mapping=endpoint.geo_mapping))
    return result