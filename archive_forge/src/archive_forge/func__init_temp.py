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
def _init_temp(self):
    self._incremental_dict = collections.defaultdict(lambda: 0)
    if self.head:
        self._ray_params.update_if_absent(temp_dir=ray._private.utils.get_ray_temp_dir())
        self._temp_dir = self._ray_params.temp_dir
    elif self._ray_params.temp_dir is None:
        assert not self._default_worker
        temp_dir = ray._private.utils.internal_kv_get_with_retry(self.get_gcs_client(), 'temp_dir', ray_constants.KV_NAMESPACE_SESSION, num_retries=ray_constants.NUM_REDIS_GET_RETRIES)
        self._temp_dir = ray._private.utils.decode(temp_dir)
    else:
        self._temp_dir = self._ray_params.temp_dir
    try_to_create_directory(self._temp_dir)
    if self.head:
        self._session_dir = os.path.join(self._temp_dir, self._session_name)
    elif self._temp_dir is None or self._session_name is None:
        assert not self._default_worker
        session_dir = ray._private.utils.internal_kv_get_with_retry(self.get_gcs_client(), 'session_dir', ray_constants.KV_NAMESPACE_SESSION, num_retries=ray_constants.NUM_REDIS_GET_RETRIES)
        self._session_dir = ray._private.utils.decode(session_dir)
    else:
        self._session_dir = os.path.join(self._temp_dir, self._session_name)
    session_symlink = os.path.join(self._temp_dir, ray_constants.SESSION_LATEST)
    try_to_create_directory(self._session_dir)
    try_to_symlink(session_symlink, self._session_dir)
    self._sockets_dir = os.path.join(self._session_dir, 'sockets')
    try_to_create_directory(self._sockets_dir)
    self._logs_dir = os.path.join(self._session_dir, 'logs')
    try_to_create_directory(self._logs_dir)
    old_logs_dir = os.path.join(self._logs_dir, 'old')
    try_to_create_directory(old_logs_dir)
    self._runtime_env_dir = os.path.join(self._session_dir, self._ray_params.runtime_env_dir_name)
    try_to_create_directory(self._runtime_env_dir)