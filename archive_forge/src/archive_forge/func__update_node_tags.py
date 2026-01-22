import logging
import random
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
from ray.autoscaler._private.aliyun.config import (
from ray.autoscaler._private.aliyun.utils import AcsClient
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.constants import BOTO_MAX_RETRIES
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def _update_node_tags(self):
    batch_updates = defaultdict(list)
    for node_id, tags in self.tag_cache_pending.items():
        for x in tags.items():
            batch_updates[x].append(node_id)
        self.tag_cache[node_id] = tags
    self.tag_cache_pending = defaultdict(dict)
    self._create_tags(batch_updates)