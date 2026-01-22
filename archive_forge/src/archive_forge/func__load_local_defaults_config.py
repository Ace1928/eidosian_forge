import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _load_local_defaults_config():
    import ray.autoscaler.local as ray_local
    return os.path.join(os.path.dirname(ray_local.__file__), 'defaults.yaml')