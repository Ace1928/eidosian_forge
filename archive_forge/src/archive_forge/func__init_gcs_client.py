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
def _init_gcs_client(self):
    if self.head:
        gcs_process = self.all_processes[ray_constants.PROCESS_TYPE_GCS_SERVER][0].process
    else:
        gcs_process = None
    for _ in range(ray_constants.NUM_REDIS_GET_RETRIES):
        gcs_address = None
        last_ex = None
        try:
            gcs_address = self.gcs_address
            client = GcsClient(address=gcs_address, cluster_id=self._ray_params.cluster_id)
            self.cluster_id = client.get_cluster_id()
            if self.head:
                client.internal_kv_get(b'dummy', None)
            self._gcs_client = client
            break
        except Exception:
            if gcs_process is not None and gcs_process.poll() is not None:
                break
            last_ex = traceback.format_exc()
            logger.debug(f'Connecting to GCS: {last_ex}')
            time.sleep(1)
    if self._gcs_client is None:
        if hasattr(self, '_logs_dir'):
            with open(os.path.join(self._logs_dir, 'gcs_server.err')) as err:
                errors = [e for e in err.readlines() if ' C ' in e or ' E ' in e][-10:]
            error_msg = '\n' + ''.join(errors) + '\n'
            raise RuntimeError(f'Failed to {('start' if self.head else 'connect to')} GCS.  Last {len(errors)} lines of error files:{error_msg}.Please check {os.path.join(self._logs_dir, 'gcs_server.out')} for details')
        else:
            raise RuntimeError(f'Failed to {('start' if self.head else 'connect to')} GCS.')
    ray.experimental.internal_kv._initialize_internal_kv(self._gcs_client)