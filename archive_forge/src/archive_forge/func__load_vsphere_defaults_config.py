import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _load_vsphere_defaults_config():
    import ray.autoscaler.vsphere as ray_vsphere
    return os.path.join(os.path.dirname(ray_vsphere.__file__), 'defaults.yaml')