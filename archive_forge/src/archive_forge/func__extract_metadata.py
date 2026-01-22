import json
import logging
from pathlib import Path
from threading import RLock
from uuid import uuid4
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resource.resources.models import DeploymentMode
from ray.autoscaler._private._azure.config import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def _extract_metadata(self, vm):
    metadata = {'name': vm.name, 'tags': vm.tags, 'status': ''}
    resource_group = self.provider_config['resource_group']
    instance = self.compute_client.virtual_machines.instance_view(resource_group_name=resource_group, vm_name=vm.name).as_dict()
    for status in instance['statuses']:
        code, state = status['code'].split('/')
        if code == 'PowerState':
            metadata['status'] = state
            break
    nic_id = vm.network_profile.network_interfaces[0].id
    metadata['nic_name'] = nic_id.split('/')[-1]
    nic = self.network_client.network_interfaces.get(resource_group_name=resource_group, network_interface_name=metadata['nic_name'])
    ip_config = nic.ip_configurations[0]
    if not self.provider_config.get('use_internal_ips', False):
        public_ip_id = ip_config.public_ip_address.id
        metadata['public_ip_name'] = public_ip_id.split('/')[-1]
        public_ip = self.network_client.public_ip_addresses.get(resource_group_name=resource_group, public_ip_address_name=metadata['public_ip_name'])
        metadata['external_ip'] = public_ip.ip_address
    metadata['internal_ip'] = ip_config.private_ip_address
    return metadata