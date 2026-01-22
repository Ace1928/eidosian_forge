from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _camel_to_snake
class AzureRMIoTHubFacts(AzureRMModuleBase):
    """Utility class to get IoT Hub facts"""

    def __init__(self):
        self.module_args = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'), show_stats=dict(type='bool'), show_quota_metrics=dict(type='bool'), show_endpoint_health=dict(type='bool'), list_keys=dict(type='bool'), test_route_message=dict(type='str'), list_consumer_groups=dict(type='bool'))
        self.results = dict(changed=False, azure_iothubs=[])
        self.name = None
        self.resource_group = None
        self.tags = None
        self.show_stats = None
        self.show_quota_metrics = None
        self.show_endpoint_health = None
        self.list_keys = None
        self.test_route_message = None
        self.list_consumer_groups = None
        super(AzureRMIoTHubFacts, self).__init__(derived_arg_spec=self.module_args, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in self.module_args:
            setattr(self, key, kwargs[key])
        response = []
        if self.name:
            response = self.get_item()
        elif self.resource_group:
            response = self.list_by_resource_group()
        else:
            response = self.list_all()
        self.results['iothubs'] = [self.to_dict(x) for x in response if self.has_tags(x.tags, self.tags)]
        return self.results

    def get_item(self):
        """Get a single IoT Hub"""
        self.log('Get properties for {0}'.format(self.name))
        item = None
        try:
            item = self.IoThub_client.iot_hub_resource.get(self.resource_group, self.name)
            return [item]
        except Exception as exc:
            self.fail('Error when getting IoT Hub {0}: {1}'.format(self.name, exc.message or str(exc)))

    def list_all(self):
        """Get all IoT Hubs"""
        self.log('List all IoT Hubs')
        try:
            return self.IoThub_client.iot_hub_resource.list_by_subscription()
        except Exception as exc:
            self.fail('Failed to list all IoT Hubs - {0}'.format(str(exc)))

    def list_by_resource_group(self):
        try:
            return self.IoThub_client.iot_hub_resource.list_by_resource_group(self.resource_group)
        except Exception as exc:
            self.fail('Failed to list IoT Hub in resource group {0} - {1}'.format(self.resource_group, exc.message or str(exc)))

    def show_hub_stats(self, resource_group, name):
        try:
            return self.IoThub_client.iot_hub_resource.get_stats(resource_group, name).as_dict()
        except Exception as exc:
            self.fail('Failed to getting statistics for IoT Hub {0}/{1}: {2}'.format(resource_group, name, str(exc)))

    def show_hub_quota_metrics(self, resource_group, name):
        result = []
        try:
            resp = self.IoThub_client.iot_hub_resource.get_quota_metrics(resource_group, name)
            while True:
                result.append(resp.next().as_dict())
        except StopIteration:
            pass
        except Exception as exc:
            self.fail('Failed to getting quota metrics for IoT Hub {0}/{1}: {2}'.format(resource_group, name, str(exc)))
        return result

    def show_hub_endpoint_health(self, resource_group, name):
        result = []
        try:
            resp = self.IoThub_client.iot_hub_resource.get_endpoint_health(resource_group, name)
            while True:
                result.append(resp.next().as_dict())
        except StopIteration:
            pass
        except Exception as exc:
            self.fail('Failed to getting health for IoT Hub {0}/{1} routing endpoint: {2}'.format(resource_group, name, str(exc)))
        return result

    def test_all_routes(self, resource_group, name):
        try:
            return self.IoThub_client.iot_hub_resource.test_all_routes(self.test_route_message, resource_group, name).routes.as_dict()
        except Exception as exc:
            self.fail('Failed to getting statistics for IoT Hub {0}/{1}: {2}'.format(resource_group, name, str(exc)))

    def list_hub_keys(self, resource_group, name):
        result = []
        try:
            resp = self.IoThub_client.iot_hub_resource.list_keys(resource_group, name)
            while True:
                result.append(resp.next().as_dict())
        except StopIteration:
            pass
        except Exception as exc:
            self.fail('Failed to getting health for IoT Hub {0}/{1} routing endpoint: {2}'.format(resource_group, name, str(exc)))
        return result

    def list_event_hub_consumer_groups(self, resource_group, name, event_hub_endpoint='events'):
        result = []
        try:
            resp = self.IoThub_client.iot_hub_resource.list_event_hub_consumer_groups(resource_group, name, event_hub_endpoint)
            while True:
                cg = resp.next()
                result.append(dict(id=cg.id, name=cg.name))
        except StopIteration:
            pass
        except Exception as exc:
            self.fail('Failed to listing consumer group for IoT Hub {0}/{1} routing endpoint: {2}'.format(resource_group, name, str(exc)))
        return result

    def route_to_dict(self, route):
        return dict(name=route.name, source=_camel_to_snake(route.source), endpoint_name=route.endpoint_names[0], enabled=route.is_enabled, condition=route.condition)

    def instance_dict_to_dict(self, instance_dict):
        result = dict()
        for key in instance_dict.keys():
            result[key] = instance_dict[key].as_dict()
        return result

    def to_dict(self, hub):
        result = dict()
        properties = hub.properties
        result['id'] = hub.id
        result['name'] = hub.name
        result['resource_group'] = parse_resource_id(hub.id).get('resource_group')
        result['location'] = hub.location
        result['tags'] = hub.tags
        result['unit'] = hub.sku.capacity
        result['sku'] = hub.sku.name.lower()
        result['cloud_to_device'] = dict(max_delivery_count=properties.cloud_to_device.feedback.max_delivery_count, ttl_as_iso8601=str(properties.cloud_to_device.feedback.ttl_as_iso8601))
        result['enable_file_upload_notifications'] = properties.enable_file_upload_notifications
        result['event_hub_endpoints'] = self.instance_dict_to_dict(properties.event_hub_endpoints)
        result['host_name'] = properties.host_name
        result['ip_filters'] = [x.as_dict() for x in properties.ip_filter_rules]
        result['routing_endpoints'] = properties.routing.endpoints.as_dict()
        result['routes'] = [self.route_to_dict(x) for x in properties.routing.routes]
        result['fallback_route'] = self.route_to_dict(properties.routing.fallback_route)
        result['status'] = properties.state
        result['storage_endpoints'] = self.instance_dict_to_dict(properties.storage_endpoints)
        if self.show_stats:
            result['statistics'] = self.show_hub_stats(result['resource_group'], hub.name)
        if self.show_quota_metrics:
            result['quota_metrics'] = self.show_hub_quota_metrics(result['resource_group'], hub.name)
        if self.show_endpoint_health:
            result['endpoint_health'] = self.show_hub_endpoint_health(result['resource_group'], hub.name)
        if self.list_keys:
            result['keys'] = self.list_hub_keys(result['resource_group'], hub.name)
        if self.test_route_message:
            result['test_route_result'] = self.test_all_routes(result['resource_group'], hub.name)
        if self.list_consumer_groups:
            result['consumer_groups'] = self.list_event_hub_consumer_groups(result['resource_group'], hub.name)
        return result