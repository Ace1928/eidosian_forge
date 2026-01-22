import copy
import logging
import time
from functools import wraps
from threading import RLock
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import googleapiclient
from ray.autoscaler._private.gcp.config import (
from ray.autoscaler._private.gcp.node import GCPTPU  # noqa
from ray.autoscaler._private.gcp.node import (
from ray.autoscaler._private.gcp.tpu_command_runner import TPUCommandRunner
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
def _thread_unsafe_terminate_node(self, node_id: str):
    logger.info('NodeProvider: {}: Terminating node'.format(node_id))
    resource = self._get_resource_depending_on_node_name(node_id)
    try:
        result = resource.delete_instance(node_id=node_id)
    except googleapiclient.errors.HttpError as http_error:
        if http_error.resp.status == 404:
            logger.warning(f'Tried to delete the node with id {node_id} but it was already gone.')
            result = None
        else:
            raise http_error from None
    return result