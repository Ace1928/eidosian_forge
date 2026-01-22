import copy
import logging
import operator
import threading
import time
import traceback
from typing import Any, Dict, Optional
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.prom_metrics import AutoscalerPrometheusMetrics
from ray.autoscaler._private.util import hash_launch_conf
from ray.autoscaler.node_launch_exception import NodeLaunchException
from ray.autoscaler.tags import (
def _launch_node(self, config: Dict[str, Any], count: int, node_type: str) -> Optional[Dict]:
    if self.node_types:
        assert node_type, node_type
    launch_config = copy.deepcopy(config.get('worker_nodes', {}))
    if node_type:
        launch_config.update(config['available_node_types'][node_type]['node_config'])
    resources = copy.deepcopy(config['available_node_types'][node_type]['resources'])
    labels = copy.deepcopy(config['available_node_types'][node_type].get('labels', {}))
    launch_hash = hash_launch_conf(launch_config, config['auth'])
    node_config = copy.deepcopy(config.get('worker_nodes', {}))
    node_tags = {TAG_RAY_NODE_NAME: 'ray-{}-worker'.format(config['cluster_name']), TAG_RAY_NODE_KIND: NODE_KIND_WORKER, TAG_RAY_NODE_STATUS: STATUS_UNINITIALIZED, TAG_RAY_LAUNCH_CONFIG: launch_hash}
    if node_type:
        node_tags[TAG_RAY_USER_NODE_TYPE] = node_type
        node_config.update(launch_config)
    node_launch_start_time = time.time()
    error_msg = None
    full_exception = None
    created_nodes = {}
    try:
        created_nodes = self.provider.create_node_with_resources_and_labels(node_config, node_tags, count, resources, labels)
    except NodeLaunchException as node_launch_exception:
        self.node_provider_availability_tracker.update_node_availability(node_type, int(node_launch_start_time), node_launch_exception)
        if node_launch_exception.src_exc_info is not None:
            full_exception = '\n'.join(traceback.format_exception(*node_launch_exception.src_exc_info))
        error_msg = f'Failed to launch {{}} node(s) of type {node_type}. ({node_launch_exception.category}): {node_launch_exception.description}'
    except Exception:
        error_msg = f'Failed to launch {{}} node(s) of type {node_type}.'
        full_exception = traceback.format_exc()
    else:
        launch_time = time.time() - node_launch_start_time
        for _ in range(count):
            self.prom_metrics.worker_create_node_time.observe(launch_time)
        self.prom_metrics.started_nodes.inc(count)
        self.node_provider_availability_tracker.update_node_availability(node_type=node_type, timestamp=int(node_launch_start_time), node_launch_exception=None)
    if error_msg is not None:
        self.event_summarizer.add(error_msg, quantity=count, aggregate=operator.add)
        self.log(error_msg)
        self.prom_metrics.node_launch_exceptions.inc()
        self.prom_metrics.failed_create_nodes.inc(count)
    else:
        self.log('Launching {} nodes, type {}.'.format(count, node_type))
        self.event_summarizer.add('Adding {} node(s) of type ' + str(node_type) + '.', quantity=count, aggregate=operator.add)
    if full_exception is not None:
        self.log(full_exception)
    return created_nodes