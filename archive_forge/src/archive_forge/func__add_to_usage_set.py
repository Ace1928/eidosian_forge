import json
import logging
import threading
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.usage.usage_constants as usage_constant
from ray.experimental.internal_kv import _internal_kv_initialized, _internal_kv_put
from ray.core.generated import usage_pb2, gcs_pb2
def _add_to_usage_set(set_name: str, value: str):
    assert _internal_kv_initialized()
    try:
        _internal_kv_put(f'{set_name}{value}'.encode(), b'', namespace=usage_constant.USAGE_STATS_NAMESPACE.encode())
    except Exception as e:
        logger.debug(f'Failed to add {value} to usage set {set_name}, {e}')