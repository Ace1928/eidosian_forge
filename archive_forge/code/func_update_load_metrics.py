import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Union
import ray
import ray._private.ray_constants as ray_constants
import ray._private.utils
from ray._private.event.event_logger import get_event_logger
from ray._private.ray_logging import setup_component_logger
from ray._raylet import GcsClient
from ray.autoscaler._private.autoscaler import StandardAutoscaler
from ray.autoscaler._private.commands import teardown_cluster
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_summarizer import EventSummarizer
from ray.autoscaler._private.load_metrics import LoadMetrics
from ray.autoscaler._private.prom_metrics import AutoscalerPrometheusMetrics
from ray.autoscaler._private.util import format_readonly_node_type
from ray.core.generated import gcs_pb2
from ray.core.generated.event_pb2 import Event as RayEvent
from ray.experimental.internal_kv import (
def update_load_metrics(self):
    """Fetches resource usage data from GCS and updates load metrics."""
    response = self.gcs_client.get_all_resource_usage(timeout=60)
    resources_batch_data = response.resource_usage_data
    log_resource_batch_data_if_desired(resources_batch_data)
    if self.readonly_config:
        new_nodes = []
        for msg in list(resources_batch_data.batch):
            node_id = msg.node_id.hex()
            new_nodes.append((node_id, msg.node_manager_address))
        self.autoscaler.provider._set_nodes(new_nodes)
    mirror_node_types = {}
    cluster_full = False
    if hasattr(response, 'cluster_full_of_actors_detected_by_gcs') and response.cluster_full_of_actors_detected_by_gcs:
        cluster_full = True
    for resource_message in resources_batch_data.batch:
        node_id = resource_message.node_id
        if self.readonly_config:
            node_type = format_readonly_node_type(node_id.hex())
            resources = {}
            for k, v in resource_message.resources_total.items():
                resources[k] = v
            mirror_node_types[node_type] = {'resources': resources, 'node_config': {}, 'max_workers': 1}
        if hasattr(resource_message, 'cluster_full_of_actors_detected') and resource_message.cluster_full_of_actors_detected:
            cluster_full = True
        total_resources = dict(resource_message.resources_total)
        available_resources = dict(resource_message.resources_available)
        waiting_bundles, infeasible_bundles = parse_resource_demands(resources_batch_data.resource_load_by_shape)
        pending_placement_groups = list(resources_batch_data.placement_group_load.placement_group_data)
        use_node_id_as_ip = self.autoscaler is not None and self.autoscaler.config['provider'].get('use_node_id_as_ip', False)
        if use_node_id_as_ip:
            peloton_id = total_resources.get('NODE_ID_AS_RESOURCE')
            if peloton_id is not None:
                ip = str(int(peloton_id))
            else:
                ip = node_id.hex()
        else:
            ip = resource_message.node_manager_address
        self.load_metrics.update(ip, node_id, total_resources, available_resources, waiting_bundles, infeasible_bundles, pending_placement_groups, cluster_full)
    if self.readonly_config:
        self.readonly_config['available_node_types'].update(mirror_node_types)