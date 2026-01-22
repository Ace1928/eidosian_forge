import copy
from enum import Enum
from typing import Any, Dict, List
from ray.autoscaler._private.util import hash_runtime_conf, prepare_config
class Provider(Enum):
    UNKNOWN = 0
    ALIYUN = 1
    AWS = 2
    AZURE = 3
    GCP = 4
    KUBERAY = 5
    LOCAL = 6