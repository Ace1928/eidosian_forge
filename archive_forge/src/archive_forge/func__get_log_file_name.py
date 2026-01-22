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
def _get_log_file_name(self, name: str, suffix: str, unique: bool=False) -> str:
    """Generate partially randomized filenames for log files.

        Args:
            name: descriptive string for this log file.
            suffix: suffix of the file. Usually it is .out of .err.
            unique: if true, a counter will be attached to `name` to
                ensure the returned filename is not already used.

        Returns:
            A tuple of two file names for redirecting (stdout, stderr).
        """
    suffix = suffix.strip('.')
    if unique:
        filename = self._make_inc_temp(suffix=f'.{suffix}', prefix=name, directory_name=self._logs_dir)
    else:
        filename = os.path.join(self._logs_dir, f'{name}.{suffix}')
    return filename