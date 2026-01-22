import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _import_aliyun(provider_config):
    from ray.autoscaler._private.aliyun.node_provider import AliyunNodeProvider
    return AliyunNodeProvider