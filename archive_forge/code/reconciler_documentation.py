import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List
from ray.autoscaler.v2.instance_manager.instance_storage import (
from ray.autoscaler.v2.instance_manager.node_provider import NodeProvider
from ray.core.generated.instance_manager_pb2 import Instance
InstanceReconciler is responsible for reconciling the difference between
    node provider and instance storage. It is also responsible for handling
    failures.
    