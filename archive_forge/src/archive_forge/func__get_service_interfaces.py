from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
import os_service_types
from heat.common import config
from heat.engine.clients import client_plugin
from heat.engine import constraints
import heat.version
def _get_service_interfaces(self):
    interfaces = {}
    if not os_service_types:
        return interfaces
    types = os_service_types.ServiceTypes()
    for name, _ in config.list_opts():
        if not name or not name.startswith('clients_'):
            continue
        project_name = name.split('_', 1)[0]
        service_data = types.get_service_data_for_project(project_name)
        if not service_data:
            continue
        service_type = service_data['service_type']
        interfaces[service_type + '_interface'] = self._get_client_option(service_type, 'endpoint_type')
    return interfaces