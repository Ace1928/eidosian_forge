import asyncio
import sys
from copy import deepcopy
from collections import defaultdict
import concurrent.futures
from dataclasses import dataclass, field
import logging
import numpy as np
import pprint
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple, Union
from ray.util.state import list_tasks
import ray
from ray.actor import ActorHandle
from ray.util.state import list_workers
from ray._private.gcs_utils import GcsAioClient, GcsChannel
from ray.util.state.state_manager import StateDataSourceClient
from ray.dashboard.state_aggregator import (
def verify_failed_task(name: str, error_type: str, error_message: Union[str, List[str]]) -> bool:
    """
    Check if a task with 'name' has failed with the exact error type 'error_type'
    and 'error_message' in the error message.
    """
    tasks = list_tasks(filters=[('name', '=', name)], detail=True)
    assert len(tasks) == 1, tasks
    t = tasks[0]
    assert t['state'] == 'FAILED', t
    assert t['error_type'] == error_type, t
    if isinstance(error_message, str):
        error_message = [error_message]
    for msg in error_message:
        assert msg in t.get('error_message', None), t
    return True