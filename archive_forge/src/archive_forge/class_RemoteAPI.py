import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Optional
import yaml
import ray
from ray._private.dict import deep_update
from ray.autoscaler._private.fake_multi_node.node_provider import (
from ray.util.queue import Empty, Queue
class RemoteAPI:
    """Remote API to control cluster state from within cluster tasks.

    This API uses a Ray queue to interact with an execution thread on the
    host machine that will execute commands passed to the queue.

    Instances of this class can be serialized and passed to Ray remote actors
    to interact with cluster state (but they can also be used outside actors).

    The API subset is limited to specific commands.

    Args:
        queue: Ray queue to push command instructions to.

    """

    def __init__(self, queue: Queue):
        self._queue = queue

    def kill_node(self, node_id: Optional[str]=None, num: Optional[int]=None, rand: Optional[str]=None):
        self._queue.put(('kill_node', dict(node_id=node_id, num=num, rand=rand)))