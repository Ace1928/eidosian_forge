import copy
import yaml
import json
import os
import socket
import sys
import time
import threading
import logging
import uuid
import warnings
import requests
from packaging.version import Version
from typing import Optional, Dict, Tuple, Type
import ray
import ray._private.services
from ray.autoscaler._private.spark.node_provider import HEAD_NODE_ID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray._private.storage import _load_class
from .utils import (
from .start_hook_base import RayOnSparkStartHook
from .databricks_hook import DefaultDatabricksRayOnSparkStartHook
def _append_default_spilling_dir_config(head_node_options, object_spilling_dir):
    if 'system_config' not in head_node_options:
        head_node_options['system_config'] = {}
    sys_conf = head_node_options['system_config']
    if 'object_spilling_config' not in sys_conf:
        sys_conf['object_spilling_config'] = json.dumps({'type': 'filesystem', 'params': {'directory_path': object_spilling_dir}})
    return head_node_options