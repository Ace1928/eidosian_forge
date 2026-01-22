import json
import logging
from collections import defaultdict
from typing import Set
from ray._private.protobuf_compat import message_to_dict
import ray
from ray._private.client_mode_hook import client_mode_hook
from ray._private.resource_spec import NODE_ID_PREFIX, HEAD_NODE_RESOURCE_NAME
from ray._private.utils import (
from ray._raylet import GlobalStateAccessor
from ray.core.generated import common_pb2
from ray.core.generated import gcs_pb2
from ray.util.annotations import DeveloperAPI
def _available_resources_per_node(self):
    """Returns a dictionary mapping node id to avaiable resources."""
    self._check_connected()
    available_resources_by_id = {}
    all_available_resources = self.global_state_accessor.get_all_available_resources()
    for available_resource in all_available_resources:
        message = gcs_pb2.AvailableResources.FromString(available_resource)
        dynamic_resources = {}
        for resource_id, capacity in message.resources_available.items():
            dynamic_resources[resource_id] = capacity
        node_id = ray._private.utils.binary_to_hex(message.node_id)
        available_resources_by_id[node_id] = dynamic_resources
    node_ids = self._live_node_ids()
    for node_id in list(available_resources_by_id.keys()):
        if node_id not in node_ids:
            del available_resources_by_id[node_id]
    return available_resources_by_id