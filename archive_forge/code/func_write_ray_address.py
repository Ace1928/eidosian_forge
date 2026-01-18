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
def write_ray_address(ray_address: str, temp_dir: Optional[str]=None):
    address_file = get_ray_address_file(temp_dir)
    if os.path.exists(address_file):
        with open(address_file, 'r') as f:
            prev_address = f.read()
        if prev_address == ray_address:
            return
        logger.info(f'Overwriting previous Ray address ({prev_address}). Running ray.init() on this node will now connect to the new instance at {ray_address}. To override this behavior, pass address={prev_address} to ray.init().')
    with open(address_file, 'w+') as f:
        f.write(ray_address)