import base64
import collections
import errno
import io
import json
import logging
import mmap
import multiprocessing
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, IO, AnyStr
import psutil
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
from ray._raylet import GcsClient, GcsClientOptions
from ray.core.generated.common_pb2 import Language
from ray._private.ray_constants import RAY_NODE_IP_FILENAME
def _find_address_from_flag(flag: str):
    """
    Attempts to find all valid Ray addresses on this node, specified by the
    flag.

    Params:
        flag: `--redis-address` or `--gcs-address`
    Returns:
        Set of detected addresses.
    """
    pids = psutil.pids()
    addresses = set()
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            cmdline = proc.cmdline()
            if len(cmdline) > 0 and 'raylet' in os.path.basename(cmdline[0]):
                for arglist in cmdline:
                    for arg in arglist.split(' '):
                        if arg.startswith(flag):
                            proc_addr = arg.split('=')[1]
                            if proc_addr != '' and proc_addr != 'None':
                                addresses.add(proc_addr)
        except psutil.AccessDenied:
            pass
        except psutil.NoSuchProcess:
            pass
    return addresses