import asyncio
import binascii
from collections import defaultdict
import contextlib
import errno
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, unquote, urlparse, parse_qsl, urlunparse
import warnings
from inspect import signature
from pathlib import Path
from subprocess import list2cmdline
from typing import (
import psutil
from google.protobuf import json_format
import ray
import ray._private.ray_constants as ray_constants
from ray.core.generated.runtime_env_common_pb2 import (
def _get_docker_cpus(cpu_quota_file_name='/sys/fs/cgroup/cpu/cpu.cfs_quota_us', cpu_period_file_name='/sys/fs/cgroup/cpu/cpu.cfs_period_us', cpuset_file_name='/sys/fs/cgroup/cpuset/cpuset.cpus', cpu_max_file_name='/sys/fs/cgroup/cpu.max') -> Optional[float]:
    cpu_quota = None
    if os.path.exists(cpu_quota_file_name) and os.path.exists(cpu_period_file_name):
        try:
            with open(cpu_quota_file_name, 'r') as quota_file, open(cpu_period_file_name, 'r') as period_file:
                cpu_quota = float(quota_file.read()) / float(period_file.read())
        except Exception:
            logger.exception('Unexpected error calculating docker cpu quota.')
    elif os.path.exists(cpu_max_file_name):
        try:
            max_file = open(cpu_max_file_name).read()
            quota_str, period_str = max_file.split()
            if quota_str.isnumeric() and period_str.isnumeric():
                cpu_quota = float(quota_str) / float(period_str)
            else:
                cpu_quota = None
        except Exception:
            logger.exception('Unexpected error calculating docker cpu quota.')
    if cpu_quota is not None and cpu_quota < 0:
        cpu_quota = None
    elif cpu_quota == 0:
        cpu_quota = 1
    cpuset_num = None
    if os.path.exists(cpuset_file_name):
        try:
            with open(cpuset_file_name) as cpuset_file:
                ranges_as_string = cpuset_file.read()
                ranges = ranges_as_string.split(',')
                cpu_ids = []
                for num_or_range in ranges:
                    if '-' in num_or_range:
                        start, end = num_or_range.split('-')
                        cpu_ids.extend(list(range(int(start), int(end) + 1)))
                    else:
                        cpu_ids.append(int(num_or_range))
                cpuset_num = len(cpu_ids)
        except Exception:
            logger.exception('Unexpected error calculating docker cpuset ids.')
    if cpu_quota and cpuset_num:
        return min(cpu_quota, cpuset_num)
    return cpu_quota or cpuset_num