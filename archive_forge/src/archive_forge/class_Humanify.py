import datetime
import json
import logging
import sys
from abc import ABC
from dataclasses import asdict, field, fields
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import ray.dashboard.utils as dashboard_utils
from ray._private.ray_constants import env_integer
from ray.core.generated.common_pb2 import TaskStatus, TaskType
from ray.core.generated.gcs_pb2 import TaskEvents
from ray.util.state.custom_types import (
from ray.util.state.exception import RayStateApiException
from ray.dashboard.modules.job.pydantic_models import JobDetails
from ray._private.pydantic_compat import IS_PYDANTIC_2
class Humanify:
    """A class containing default methods to
    convert units into a human readable string."""

    def timestamp(x: float):
        """Converts miliseconds to a datetime object."""
        return str(datetime.datetime.fromtimestamp(x / 1000))

    def memory(x: int):
        """Converts raw bytes to a human readable memory size."""
        if x >= 2 ** 30:
            return str(format(x / 2 ** 30, '.3f')) + ' GiB'
        elif x >= 2 ** 20:
            return str(format(x / 2 ** 20, '.3f')) + ' MiB'
        elif x >= 2 ** 10:
            return str(format(x / 2 ** 10, '.3f')) + ' KiB'
        return str(format(x, '.3f')) + ' B'

    def duration(x: int):
        """Converts miliseconds to a human readable duration."""
        return str(datetime.timedelta(milliseconds=x))

    def events(events: List[dict]):
        """Converts a list of task events into a human readable format."""
        for event in events:
            if 'created_ms' in event:
                event['created_ms'] = Humanify.timestamp(event['created_ms'])
        return events

    def node_resources(resources: dict):
        """Converts a node's resources into a human readable format."""
        for resource in resources:
            if 'memory' in resource:
                resources[resource] = Humanify.memory(resources[resource])
        return resources