import atexit
import collections
import datetime
import errno
import json
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from typing import Dict, Optional, Tuple, IO, AnyStr
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services
from ray._private import storage
from ray._raylet import GcsClient, get_session_key_from_storage
from ray._private.resource_spec import ResourceSpec
from ray._private.services import serialize_config, get_address
from ray._private.utils import open_log, try_to_create_directory, try_to_symlink
def _write_cluster_info_to_kv(self):
    """Write the cluster metadata to GCS.
        Cluster metadata is always recorded, but they are
        not reported unless usage report is enabled.
        Check `usage_stats_head.py` for more details.
        """
    import ray._private.usage.usage_lib as ray_usage_lib
    ray_usage_lib.put_cluster_metadata(self.get_gcs_client())
    added = self.get_gcs_client().internal_kv_put(b'session_name', self._session_name.encode(), False, ray_constants.KV_NAMESPACE_SESSION)
    if not added:
        curr_val = self.get_gcs_client().internal_kv_get(b'session_name', ray_constants.KV_NAMESPACE_SESSION)
        assert curr_val == self._session_name.encode('utf-8'), f'Session name {self._session_name} does not match persisted value {curr_val}. Perhaps there was an error connecting to Redis.'
    self.get_gcs_client().internal_kv_put(b'session_dir', self._session_dir.encode(), True, ray_constants.KV_NAMESPACE_SESSION)
    self.get_gcs_client().internal_kv_put(b'temp_dir', self._temp_dir.encode(), True, ray_constants.KV_NAMESPACE_SESSION)
    if self._ray_params.storage is not None:
        self.get_gcs_client().internal_kv_put(b'storage', self._ray_params.storage.encode(), True, ray_constants.KV_NAMESPACE_SESSION)
    if self._ray_params.tracing_startup_hook:
        self.get_gcs_client().internal_kv_put(b'tracing_startup_hook', self._ray_params.tracing_startup_hook.encode(), True, ray_constants.KV_NAMESPACE_TRACING)