import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _import_external(provider_config):
    provider_cls = load_function_or_class(path=provider_config['module'])
    return provider_cls