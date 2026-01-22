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
@synchronized
def _get_filtered_nodes(self, tag_filters):
    cluster_tag_filters = {**tag_filters, TAG_RAY_CLUSTER_NAME: self.cluster_name}

    def match_tags(vm):
        for k, v in cluster_tag_filters.items():
            if vm.tags.get(k) != v:
                return False
        return True
    vms = self.compute_client.virtual_machines.list(resource_group_name=self.provider_config['resource_group'])
    nodes = [self._extract_metadata(vm) for vm in filter(match_tags, vms)]
    self.cached_nodes = {node['name']: node for node in nodes}
    return self.cached_nodes