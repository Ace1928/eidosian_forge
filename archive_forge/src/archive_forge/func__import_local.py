import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _import_local(provider_config):
    if 'coordinator_address' in provider_config:
        from ray.autoscaler._private.local.coordinator_node_provider import CoordinatorSenderNodeProvider
        return CoordinatorSenderNodeProvider
    else:
        from ray.autoscaler._private.local.node_provider import LocalNodeProvider
        return LocalNodeProvider