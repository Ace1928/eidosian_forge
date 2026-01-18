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