from API responses.
import abc
import logging
import re
import time
from collections import UserDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError
from ray.autoscaler.tags import TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_NAME
class GCPCompute(GCPResource):
    """Abstraction around GCP compute resource"""

    def wait_for_operation(self, operation: dict, max_polls: int=MAX_POLLS, poll_interval: int=POLL_INTERVAL) -> dict:
        """Poll for compute zone operation until finished."""
        logger.info(f'wait_for_compute_zone_operation: Waiting for operation {operation['name']} to finish...')
        for _ in range(max_polls):
            result = self.resource.zoneOperations().get(project=self.project_id, operation=operation['name'], zone=self.availability_zone).execute()
            if 'error' in result:
                raise Exception(result['error'])
            if result['status'] == 'DONE':
                logger.info(f'wait_for_compute_zone_operation: Operation {operation['name']} finished.')
                break
            time.sleep(poll_interval)
        return result

    def list_instances(self, label_filters: Optional[dict]=None, is_terminated: bool=False) -> List[GCPComputeNode]:
        label_filters = label_filters or {}
        if label_filters:
            label_filter_expr = '(' + ' AND '.join(['(labels.{key} = {value})'.format(key=key, value=value) for key, value in label_filters.items()]) + ')'
        else:
            label_filter_expr = ''
        statuses = GCPComputeNode.TERMINATED_STATUSES if is_terminated else GCPComputeNode.NON_TERMINATED_STATUSES
        instance_state_filter_expr = '(' + ' OR '.join(['(status = {status})'.format(status=status) for status in statuses]) + ')'
        cluster_name_filter_expr = '(labels.{key} = {value})'.format(key=TAG_RAY_CLUSTER_NAME, value=self.cluster_name)
        tpu_negation_filter_expr = '(NOT labels.{label}:*)'.format(label='tpu_cores')
        not_empty_filters = [f for f in [label_filter_expr, instance_state_filter_expr, cluster_name_filter_expr, tpu_negation_filter_expr] if f]
        filter_expr = ' AND '.join(not_empty_filters)
        response = self.resource.instances().list(project=self.project_id, zone=self.availability_zone, filter=filter_expr).execute()
        instances = response.get('items', [])
        return [GCPComputeNode(i, self) for i in instances]

    def get_instance(self, node_id: str) -> GCPComputeNode:
        instance = self.resource.instances().get(project=self.project_id, zone=self.availability_zone, instance=node_id).execute()
        return GCPComputeNode(instance, self)

    def set_labels(self, node: GCPComputeNode, labels: dict, wait_for_operation: bool=True) -> dict:
        body = {'labels': dict(node['labels'], **labels), 'labelFingerprint': node['labelFingerprint']}
        node_id = node['name']
        operation = self.resource.instances().setLabels(project=self.project_id, zone=self.availability_zone, instance=node_id, body=body).execute()
        if wait_for_operation:
            result = self.wait_for_operation(operation)
        else:
            result = operation
        return result

    def _convert_resources_to_urls(self, configuration_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures that resources are in their full URL form.

        GCP expects machineType and acceleratorType to be a full URL (e.g.
        `zones/us-west1/machineTypes/n1-standard-2`) instead of just the
        type (`n1-standard-2`)

        Args:
            configuration_dict: Dict of options that will be passed to GCP
        Returns:
            Input dictionary, but with possibly expanding `machineType` and
                `acceleratorType`.
        """
        configuration_dict = deepcopy(configuration_dict)
        existing_machine_type = configuration_dict['machineType']
        if not re.search('.*/machineTypes/.*', existing_machine_type):
            configuration_dict['machineType'] = 'zones/{zone}/machineTypes/{machine_type}'.format(zone=self.availability_zone, machine_type=configuration_dict['machineType'])
        for accelerator in configuration_dict.get('guestAccelerators', []):
            gpu_type = accelerator['acceleratorType']
            if not re.search('.*/acceleratorTypes/.*', gpu_type):
                accelerator['acceleratorType'] = 'projects/{project}/zones/{zone}/acceleratorTypes/{accelerator}'.format(project=self.project_id, zone=self.availability_zone, accelerator=gpu_type)
        return configuration_dict

    def create_instance(self, base_config: dict, labels: dict, wait_for_operation: bool=True) -> Tuple[dict, str]:
        config = self._convert_resources_to_urls(base_config)
        config.pop('networkConfig', None)
        name = _generate_node_name(labels, GCPNodeType.COMPUTE.value)
        labels = dict(config.get('labels', {}), **labels)
        config.update({'labels': dict(labels, **{TAG_RAY_CLUSTER_NAME: self.cluster_name}), 'name': name})
        source_instance_template = config.pop('sourceInstanceTemplate', None)
        operation = self.resource.instances().insert(project=self.project_id, zone=self.availability_zone, sourceInstanceTemplate=source_instance_template, body=config).execute()
        if wait_for_operation:
            result = self.wait_for_operation(operation)
        else:
            result = operation
        return (result, name)

    def delete_instance(self, node_id: str, wait_for_operation: bool=True) -> dict:
        operation = self.resource.instances().delete(project=self.project_id, zone=self.availability_zone, instance=node_id).execute()
        if wait_for_operation:
            result = self.wait_for_operation(operation)
        else:
            result = operation
        return result

    def stop_instance(self, node_id: str, wait_for_operation: bool=True) -> dict:
        operation = self.resource.instances().stop(project=self.project_id, zone=self.availability_zone, instance=node_id).execute()
        if wait_for_operation:
            result = self.wait_for_operation(operation)
        else:
            result = operation
        return result

    def start_instance(self, node_id: str, wait_for_operation: bool=True) -> dict:
        operation = self.resource.instances().start(project=self.project_id, zone=self.availability_zone, instance=node_id).execute()
        if wait_for_operation:
            result = self.wait_for_operation(operation)
        else:
            result = operation
        return result