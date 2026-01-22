import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _load_gcp_defaults_config():
    import ray.autoscaler.gcp as ray_gcp
    return os.path.join(os.path.dirname(ray_gcp.__file__), 'defaults.yaml')