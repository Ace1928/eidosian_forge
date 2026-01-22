import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
def _import_gcp(provider_config):
    try:
        import googleapiclient
    except ImportError as e:
        raise ImportError('The Ray GCP VM launcher requires the Google API Client to be installed. You can install it with `pip install google-api-python-client`.') from e
    from ray.autoscaler._private.gcp.node_provider import GCPNodeProvider
    return GCPNodeProvider