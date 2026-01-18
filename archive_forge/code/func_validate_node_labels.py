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
def validate_node_labels(labels: Dict[str, str]):
    if labels is None:
        return
    for key in labels.keys():
        if key.startswith(ray_constants.RAY_DEFAULT_LABEL_KEYS_PREFIX):
            raise ValueError(f'Custom label keys `{key}` cannot start with the prefix `{ray_constants.RAY_DEFAULT_LABEL_KEYS_PREFIX}`. This is reserved for Ray defined labels.')