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
class GCPTPU(GCPResource):
    """Abstraction around GCP TPU resource"""

    @property
    def path(self):
        return f'projects/{self.project_id}/locations/{self.availability_zone}'

    def wait_for_operation(self, operation: dict, max_polls: int=MAX_POLLS_TPU, poll_interval: int=POLL_INTERVAL) -> dict:
        """Poll for TPU operation until finished."""
        logger.info(f'wait_for_tpu_operation: Waiting for operation {operation['name']} to finish...')
        for _ in range(max_polls):
            result = self.resource.projects().locations().operations().get(name=f'{operation['name']}').execute()
            if 'error' in result:
                raise Exception(result['error'])
            if 'response' in result:
                logger.info(f'wait_for_tpu_operation: Operation {operation['name']} finished.')
                break
            time.sleep(poll_interval)
        return result

    def list_instances(self, label_filters: Optional[dict]=None, is_terminated: bool=False) -> List[GCPTPUNode]:
        response = self.resource.projects().locations().nodes().list(parent=self.path).execute()
        instances = response.get('nodes', [])
        instances = [GCPTPUNode(i, self) for i in instances]
        label_filters = label_filters or {}
        label_filters[TAG_RAY_CLUSTER_NAME] = self.cluster_name

        def filter_instance(instance: GCPTPUNode) -> bool:
            if instance.is_terminated():
                return False
            labels = instance.get_labels()
            if label_filters:
                for key, value in label_filters.items():
                    if key not in labels:
                        return False
                    if value != labels[key]:
                        return False
            return True
        instances = list(filter(filter_instance, instances))
        return instances

    def get_instance(self, node_id: str) -> GCPTPUNode:
        instance = self.resource.projects().locations().nodes().get(name=node_id).execute()
        return GCPTPUNode(instance, self)

    @_retry_on_exception(HttpError, 'unable to queue the operation')
    def set_labels(self, node: GCPTPUNode, labels: dict, wait_for_operation: bool=True) -> dict:
        body = {'labels': dict(node['labels'], **labels)}
        update_mask = 'labels'
        operation = self.resource.projects().locations().nodes().patch(name=node['name'], updateMask=update_mask, body=body).execute()
        if wait_for_operation:
            result = self.wait_for_operation(operation)
        else:
            result = operation
        return result

    def create_instance(self, base_config: dict, labels: dict, wait_for_operation: bool=True) -> Tuple[dict, str]:
        config = base_config.copy()
        config.pop('networkInterfaces', None)
        name = _generate_node_name(labels, GCPNodeType.TPU.value)
        labels = dict(config.get('labels', {}), **labels)
        config.update({'labels': dict(labels, **{TAG_RAY_CLUSTER_NAME: self.cluster_name})})
        if 'networkConfig' not in config:
            config['networkConfig'] = {}
        if 'enableExternalIps' not in config['networkConfig']:
            config['networkConfig']['enableExternalIps'] = True
        if 'serviceAccounts' in config:
            config['serviceAccount'] = config.pop('serviceAccounts')[0]
            config['serviceAccount']['scope'] = config['serviceAccount'].pop('scopes')
        operation = self.resource.projects().locations().nodes().create(parent=self.path, body=config, nodeId=name).execute()
        if wait_for_operation:
            result = self.wait_for_operation(operation)
        else:
            result = operation
        return (result, name)

    def delete_instance(self, node_id: str, wait_for_operation: bool=True) -> dict:
        operation = self.resource.projects().locations().nodes().delete(name=node_id).execute()
        if wait_for_operation:
            result = self.wait_for_operation(operation, max_polls=MAX_POLLS)
        else:
            result = operation
        return result

    def stop_instance(self, node_id: str, wait_for_operation: bool=True) -> dict:
        operation = self.resource.projects().locations().nodes().stop(name=node_id).execute()
        if wait_for_operation:
            result = self.wait_for_operation(operation, max_polls=MAX_POLLS)
        else:
            result = operation
        return result

    def start_instance(self, node_id: str, wait_for_operation: bool=True) -> dict:
        operation = self.resource.projects().locations().nodes().start(name=node_id).execute()
        if wait_for_operation:
            result = self.wait_for_operation(operation, max_polls=MAX_POLLS)
        else:
            result = operation
        return result