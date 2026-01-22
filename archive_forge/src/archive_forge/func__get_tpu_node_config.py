from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def _get_tpu_node_config():
    tpu_config_env = os.environ.get(_DEFAULT_TPUCONFIG_VARIABLE)
    if tpu_config_env:
        return json.loads(tpu_config_env)
    return None