import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _import_fake_multinode(provider_config):
    from ray.autoscaler._private.fake_multi_node.node_provider import FakeMultiNodeProvider
    return FakeMultiNodeProvider