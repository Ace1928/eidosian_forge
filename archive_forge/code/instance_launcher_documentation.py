import logging
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List
from ray.autoscaler._private.constants import (
from ray.autoscaler.v2.instance_manager.instance_storage import (
from ray.autoscaler.v2.instance_manager.node_provider import NodeProvider
from ray.core.generated.instance_manager_pb2 import Instance
Launches instances of the given type.

        Args:
            instance_type: type of instance to launch.
            instances: list of instances to launch. These instances should
                have been marked as QUEUED with instance_type set.
        Returns:
            num of instances launched.
        